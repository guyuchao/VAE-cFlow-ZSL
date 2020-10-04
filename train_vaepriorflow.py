import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import glob
import json
import argparse
import os
import random
import numpy as np
from time import gmtime, strftime
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from vaeflow.glow import Glow
import classifier
from dataset_GBU import FeatDataLayer, DATA_LOADER


import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA1', help='dataset: CUB, AWA1, AWA2, SUN')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)

parser.add_argument('--niter', type=int, default=25000, help='number of epochs to train for')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to train generater')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=496*12, help='number features to generate per class')

parser.add_argument('--resume',  type=str, help='the model to resume')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=600)
parser.add_argument('--manualSeed',default=None, type=int, help='manual seed')#25820 52.8 CUB  SUN 6942 42.8

parser.add_argument('--Knn', type=int, default=20, help='K value')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
opt = parser.parse_args()


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))

def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.y_dim = dataset.ntrain_class

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.numpy(), opt)
    result_zsl_knn = Result()
    result_gzsl_soft = Result()

    netG = Glow(classes=opt.y_dim,condition_dim=opt.C_dim).cuda()

    out_dir = 'out/{}/shuffle'.format(opt.dataset)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    if opt.resume:
        if os.pathls\
                .isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            train_z = checkpoint['latent_z'].cuda()
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    initial=True
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)

    for it in range(start_step, opt.niter+1):
        blobs = data_layer.forward()
        feat_data = blobs['data']  # image data
        labels = blobs['labels'].astype(int)  # class labels
        idx    = blobs['idx'].astype(int)

        C = np.array([dataset.train_att[i,:] for i in labels])
        L = torch.from_numpy(labels).cuda()
        C = torch.from_numpy(C.astype('float32')).cuda()
        X = torch.from_numpy(feat_data).cuda()
        X = X.view(*X.shape, 1, 1)

        if initial is True:
            netG(x=X, y_onehot=C, reverse=False)
            initial = False

        z, nll,vaeloss, y_logit = netG(x=X, y_onehot=C,reverse=False)

        loss_generative = Glow.loss_generative(nll)
        loss_classes = Glow.loss_class(y_logit, L)
        loss = loss_generative +vaeloss + loss_classes * 0.01
        netG.zero_grad()
        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()

        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; epoch: {} Gloss: {:.3f} vaeloss: {:.3f} clsloss: {:.3f}'.format(it, opt.niter, it//opt.evl_interval,float(loss_generative),float(vaeloss),float(loss_classes))
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it:
            netG.eval()
            gen_feat, gen_label = synthesize_feature_test(netG, dataset, 300,0.5,opt)
            """ ZSL"""

            acc = eval_zsl_knn(gen_feat.numpy(), gen_label.numpy(), dataset)
            result_zsl_knn.update(it, acc)
            log_print("{}nn Classifer: ".format(opt.Knn), log_dir)
            log_print("Accuracy is {:.2f}%, Best_acc [{:.2f}% | Iter-{}]".format(acc, result_zsl_knn.best_acc,
                                                                                 result_zsl_knn.best_iter), log_dir)

            gen_feat, gen_label = synthesize_feature_test(netG, dataset, opt.nSample,1.0,opt)

            """ GZSL"""
            # note test label need be shift with offset ntrain_class
            train_X = torch.cat((dataset.train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label+dataset.ntrain_class), 0)

            cls = classifier.CLASSIFIER(train_X, train_Y, dataset, dataset.ntrain_class + dataset.ntest_class,
                                             True, opt.classifier_lr, 0.5, 25, opt.nSample, True)
            result_gzsl_soft.update_gzsl(it, cls.acc_unseen, cls.acc_seen, cls.H)
            log_print("GZSL Softmax:", log_dir)
            log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)


            if result_zsl_knn.save_model:
                files2remove = glob.glob(out_dir + '/Best_model_ZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                save_model(it, netG,  opt.manualSeed, log_text,
                           out_dir + '/Best_model_ZSL_Acc_{:.2f}.tar'.format(result_zsl_knn.acc_list[-1]))


            if result_gzsl_soft.save_model:
                files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                save_model(it, netG, opt.manualSeed, log_text,
                           out_dir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl_soft.best_acc,
                                                                                             result_gzsl_soft.best_acc_S_T,
                                                                                             result_gzsl_soft.best_acc_U_T))
            netG.train()

        if it % opt.save_interval == 0 and it:
            save_model(it, netG, opt.manualSeed, log_text,
                       out_dir + '/Iter_{:d}.tar'.format(it))
            print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))

def log_print(s, log):
    print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')

def getloss(pred, x, z, opt):
    loss = 1/(2*opt.sigma**2) * torch.pow(x - pred, 2).sum() + 1/2 * torch.pow(z, 2).sum()
    loss /= x.size(0)
    return loss

def save_model(it, netG, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


def synthesize_feature_test(netG, dataset, nSample,tempture,opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * nSample, opt.X_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            text_feat = np.tile(dataset.test_att[i].astype('float32'), (nSample, 1))
            text_feat = torch.from_numpy(text_feat).cuda()
            G_sample,_=netG(z=None, y_onehot=text_feat, eps_std=tempture, reverse=True)
            G_sample=G_sample.view(*G_sample.shape[:2])
            gen_feat[i*nSample:(i+1)*nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([nSample])*i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))



def eval_zsl_knn(gen_feat, gen_label, dataset):
    # cosince predict K-nearest Neighbor
    n_test_sample = dataset.test_unseen_feature.shape[0]
    sim = cosine_similarity(dataset.test_unseen_feature, gen_feat)
    # only count first K nearest neighbor
    idx_mat = np.argsort(-1 * sim, axis=1)[:, 0:opt.Knn]
    label_mat = gen_label[idx_mat.flatten()].reshape((n_test_sample,-1))
    preds = np.zeros(n_test_sample)
    for i in range(n_test_sample):
        label_count = Counter(label_mat[i]).most_common(1)
        preds[i] = label_count[0][0]
    acc = eval_MCA(preds, dataset.test_unseen_label.numpy()) * 100
    return acc


def eval_MCA(preds, y):
    cls_label = np.unique(y)
    acc = list()
    for i in cls_label:
        acc.append((preds[y == i] == i).mean())
    return np.asarray(acc).mean()



class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.save_model = True
    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.iter_list += [it]
        self.save_model = False
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U_T = acc_u
            self.best_acc_S_T = acc_s
            self.save_model = True


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.normal_(m.weight.data,  mean=0, std=0.02)
        init.constant_(m.bias, 0.0)




if __name__ == "__main__":
    train()

