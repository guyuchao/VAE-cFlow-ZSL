import torch.nn as nn
from vaeflow.flow import FlowNet
import torch
import vaeflow.thops as thops
import numpy as np
from vaeflow.coupling import Conv2dZeros

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight,gain=0.5)

class Attrencoder(nn.Module):
    def __init__(self,condition_dim):
        super(Attrencoder, self).__init__()
        self.encoder=nn.Sequential(nn.Linear(condition_dim,1450),
                                   nn.ReLU(),nn.Linear(1450,2048),nn.ReLU())
        self._mu = nn.Linear(in_features=2048, out_features=2048)
        self._logvar = nn.Linear(in_features=2048, out_features=2048)
        self.apply(weights_init)

    def forward(self, x):
        h = self.encoder(x)
        mu = self._mu(h)
        logvar = self._logvar(h)
        return mu, logvar

class Attrdecoder(nn.Module):
    def __init__(self,condition_dim):
        super(Attrdecoder, self).__init__()
        self.decoder=nn.Sequential(nn.Linear(2048,665),
                                   nn.ReLU(),nn.Linear(665,condition_dim))
        self.apply(weights_init)

    def forward(self, x):
        x=x.view(*x.shape[:2])
        return self.decoder(x)


class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__(in_channels, out_channels)
        self.logscale_factor = logscale_factor
        # set logs parameter
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)

class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return thops.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

class Glow(nn.Module):
    CE = nn.CrossEntropyLoss()
    RC = nn.L1Loss(size_average=False)

    def __init__(self, learn_top=True,y_condition=True,classes=None,condition_dim=None):
        super().__init__()
        self.flow = FlowNet(image_shape=[2048,1,1],hidden_channels=512)
        self.y_classes = classes
        self.condition_dim=condition_dim
        # for prior
        self.LTop=learn_top
        self.y_condition=y_condition
        if self.LTop:
            C = self.flow.output_shapes[-1][1]
            self.learn_top = Conv2dZeros(C * 2, C * 2)
            self.C=C

        if self.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.encoder = Attrencoder(condition_dim)
            self.vaeDEC = Attrdecoder(condition_dim)
            self.project_class = LinearZeros(
                C, self.y_classes)

    def prior(self, y_onehot=None):
        B, C = y_onehot.shape[0],2*self.C
        h = torch.zeros((B,C,1,1),device=y_onehot.device)
        if self.learn_top:
            h = self.learn_top(h)
        means,logs=thops.split_feature(h, "split")

        if self.y_condition:
            assert y_onehot is not None
            y_onehot=y_onehot.view(*y_onehot.shape[:2])
            ymeans,ylogs = self.encoder(y_onehot)
            ymeans=ymeans.view(*ymeans.shape, 1, 1)
            ylogs = ylogs.view(*ylogs.shape, 1, 1)
            means+=ymeans
            logs+=ylogs
        return means,logs,ymeans,ylogs

    def forward(self, x=None, y_onehot=None, z=None,
                eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, y_onehot)
        else:
            return self.reverse_flow(z, y_onehot, eps_std)

    def reparameterize(self, mu, logvar, reparameterize_with_noise=True):
        mu=mu.view(*mu.shape[:2])
        logvar=logvar.view(*logvar.shape[:2])
        if reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu

    def normal_flow(self, x, attribute):

        self.encoder.train()
        self.vaeDEC.train()
        pixels = thops.pixels(x)
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        #z = x + torch.normal(mean=torch.zeros_like(x),std=torch.ones_like(x)/100)
        z=x
        # encode
        z, objective = self.flow(z, logdet=logdet, reverse=False)

        # prior
        mean, logs,mu_att,logvar_att = self.prior(attribute)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        att_from_att= self.vaeDEC(z_from_att)
        objective += GaussianDiag.logp(mean, logs, z)

        vaeRCloss=Glow.RC(att_from_att,attribute)


        vaeKLloss=torch.sum(1 + logvar_att - logvar_att.exp())#*0.5+0.5*torch.sum(- mu_att.pow(2))
        vaeloss=vaeRCloss-vaeKLloss

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # return
        nll = (-objective) / float(np.log(2.) * pixels)

        return z, nll, vaeloss, y_logits

    def reverse_flow(self, z, y_onehot, eps_std):
        with torch.no_grad():
            mean, logs,_,_ = self.prior(y_onehot)
            if z is None:
                z = GaussianDiag.sample(mean, logs, eps_std)
            x = self.flow(z, eps_std=eps_std, reverse=True)
        return x,z

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

    @staticmethod
    def loss_multi_classes(y_logits, y_onehot):
        if y_logits is None:
            return 0
        else:
            return Glow.BCE(y_logits, y_onehot.float())

    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            return Glow.CE(y_logits, y.long())

if __name__=="__main__":
    model=Glow(True)
    x=torch.randn((2,300,1,1))
    condition=torch.zeros((2,300,1,1))
    model(y_onehot=condition,reverse=True)
