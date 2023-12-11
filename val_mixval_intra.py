import argparse
import os
import random
import math
import os.path as osp
import copy
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import pre_process as prep
from data_list import ImageList, ImageList_w_path
from classifier import ImageClassifier, ImageClassifierMDD, ImageClassifierAFN
from backbone import get_model

def log_dset(out_path, data_list):
    if not osp.exists(out_path):
        with open(out_path, 'w') as fp:
            for item in data_list:
                fp.write("%s" % item)

def entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy 

def init_model(args):        
    backbone = get_model(args.net, pretrain=True)
    #pool_layer = nn.Identity() if args.no_pool else None
    if args.method == 'MDD':
        classifier = ImageClassifierMDD(backbone, args.class_num, bottleneck_dim=args.bottleneck_dim, width=args.width)
    elif args.method == 'SAFN':
        classifier = ImageClassifierAFN(backbone, args.class_num, bottleneck_dim=args.bottleneck_dim)
    else:
        classifier = ImageClassifier(backbone, args.class_num, bottleneck_dim=args.bottleneck_dim)
    return classifier

def load_ckpt(args):
    ckpt_path = args.ckpt_path
    ckpt_dict = torch.load(ckpt_path)
    filtered_state_dict = OrderedDict()
    for k in ckpt_dict:
        if 'backbone.fc' in k:
            pass
        else:
            filtered_state_dict[k] = ckpt_dict[k]
    return filtered_state_dict

def load_ckpt_mdd(args):
    ckpt_path = args.ckpt_path
    ckpt_dict = torch.load(ckpt_path)
    filtered_state_dict = OrderedDict()
    for k in ckpt_dict:
        if 'fc' in k or 'adv' in k:
            pass
        elif 'bottleneck.1' in k:
            newk = k.replace('bottleneck.1', 'bottleneck.0')
            filtered_state_dict[newk] = ckpt_dict[k]
        elif 'bottleneck.2' in k:
            newk = k.replace('bottleneck.2', 'bottleneck.1')
            filtered_state_dict[newk] = ckpt_dict[k]
        else:
            filtered_state_dict[k] = ckpt_dict[k]
    return filtered_state_dict

def test(config, args, net):
    prep_dict = {}
    prep_dict["test_tgt"] = prep.image_test(**config["prep"]["params"])
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    test_bs = data_config["test_tgt"]["batch_size"]
    dsets["test_tgt"] = ImageList_w_path(open(data_config["test_tgt"]["list_path"]).readlines(), transform=prep_dict["test_tgt"])
    dset_loaders["test_tgt"] = DataLoader(dsets["test_tgt"], batch_size=test_bs, shuffle=False, num_workers=4, drop_last=False)

    net = net.cuda()
    net.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["test_tgt"])
        paths = []
        for i in range(len(dset_loaders["test_tgt"])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            full_paths = data[3]
            paths += full_paths
            inputs = inputs.cuda()
            outputs = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    index_array = predict.argsort()
    paths = np.take(paths, index_array).tolist()

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])    
    all_label = all_label.long()
    mean_ent = torch.mean(entropy(nn.Softmax(dim=-1)(all_output))).item()
    log_str = "Testing accuracy: {:.4f}, mean entropy is {:.4f}.\n".format(accuracy, mean_ent)

    if config["dataset"] == "visda":
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1)
        cls_acc_str = ' '.join(['{:.4f}'.format(x) for x in acc.tolist()])
        log_str = "Testing per-class accuracy: {}, mean-class accuracy: {:.4f}, mean accuracy: {:.4f}, mean entropy is {:.4f}.\n".\
                format(cls_acc_str, np.mean(acc), accuracy, mean_ent)
    torch.save(all_output[index_array], config["task_pred_file"])
    log_dset(config["img_intra_list"], paths)
    config["total_log_file"].write(log_str)
    print(log_str)

def mixval(net, args, config):

    if osp.isfile(config["task_pred_file"]):
        saved_logits = torch.load(config["task_pred_file"])
    else:
        test(config, args, net)
        saved_logits = torch.load(config["task_pred_file"])
    raw_pl = F.one_hot(saved_logits.max(dim=-1)[1], num_classes=args.class_num).float()

    if osp.isfile(config["img_intra_list"]):
        intra_list = config["img_intra_list"]
    else:
        intra_list = None
        print('Image list for intra-cluser mixup not found!')
    
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["target"] = prep.image_test(**config["prep"]['params'])
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    dsets["target"] = ImageList(open(intra_list).readlines(), transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.bs, shuffle=False, num_workers=4, drop_last=False)
    net = net.cuda()
    net.eval()

    total = 0
    start_gather = True
    all_mix_logits = None
    all_mix_labels = None
    all_raw_logits = None
    all_raw_labels = None
    all_same_idx = None
    all_diff_idx = None

    with torch.no_grad():
        for ep in range(args.epoch):
            for batch_idx, (inputs, labels, idx) in enumerate(dset_loaders["target"]):
                batch_size = inputs.size(0)
                if batch_size < args.bs:
                    last_bs = batch_size
                inputs = inputs.cuda()
                #if args.synth == 'betamix':
                #    mix_lam = np.random.beta(args.alpha, args.alpha)
                if args.synth == 'mixup':
                    mix_lam = args.lam
                #rand_idx = torch.randperm(batch_size)
                rand_idx = torch.arange(batch_size).flip([0]).cuda()
                logit_a = saved_logits[idx]
                pl_a = raw_pl[idx]
                pl_b = pl_a[rand_idx]
                raw_labels = labels

                inputs_a = inputs
                inputs_b = inputs_a[rand_idx]
                same_idx = (pl_a.max(dim=-1)[1]==pl_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0] + total
                diff_idx = (pl_a.max(dim=-1)[1]!=pl_b.max(dim=-1)[1]).nonzero(as_tuple=True)[0] + total
                mix_inputs = mix_lam * inputs_a + (1 - mix_lam) * inputs_b
                mix_labels = mix_lam * pl_a + (1 - mix_lam) * pl_b
                mix_logits = net(mix_inputs).detach().float().cpu()

                if start_gather:
                    all_mix_logits = mix_logits
                    all_mix_labels = mix_labels
                    all_raw_logits = logit_a
                    all_raw_labels = raw_labels
                    all_same_idx = same_idx
                    all_diff_idx = diff_idx
                    start_gather = False
                else:
                    all_mix_logits = torch.cat((all_mix_logits, mix_logits), 0)
                    all_mix_labels = torch.cat((all_mix_labels, mix_labels), 0)
                    all_raw_logits = torch.cat((all_raw_logits, logit_a), 0)
                    all_raw_labels = torch.cat((all_raw_labels, raw_labels), 0)
                    all_same_idx = torch.cat((all_same_idx, same_idx), 0)
                    all_diff_idx = torch.cat((all_diff_idx, diff_idx), 0)

                total += batch_size
            
    raw_logits = all_raw_logits[:saved_logits.shape[0]]
    raw_labels = all_raw_labels[:saved_logits.shape[0]]

    # real accuracy of target-domain predictions evaluated by ground truth (gt)
    target_acc = torch.sum(raw_logits.max(dim=-1)[1].float() == raw_labels.float()).item() / float(raw_labels.size()[0])    

    # our ICE with only intra-cluster mixup: max is best
    ice_intra = torch.sum(all_mix_logits[all_same_idx].max(dim=-1)[1].float() == all_mix_labels[all_same_idx].max(dim=-1)[1].float()).item() / \
            float(all_mix_labels[all_same_idx].size()[0])

    # entropy: min is best
    raw_sfmx_p = raw_logits.softmax(dim=-1)
    raw_mean_p = raw_sfmx_p.mean(dim=0)
    raw_ent = entropy(raw_sfmx_p).mean()

    # im: max is best
    raw_div = -torch.sum(raw_mean_p * torch.log(raw_mean_p + 1e-5))
    raw_im = raw_div - raw_ent

    # corr-c: min is best
    raw_corr = torch.mm(raw_sfmx_p.t(), raw_sfmx_p)
    raw_softmaxCorr = raw_corr.diag().sum() / ((raw_corr**2).sum()**0.5)

    # snd: max is best
    raw_normalized = F.normalize(raw_logits).cpu()
    raw_mat = torch.matmul(raw_normalized, raw_normalized.t()) / 0.05
    raw_mask = torch.eye(raw_mat.size(0), raw_mat.size(0)).bool()
    raw_mat.masked_fill_(raw_mask, -1 / 0.05)
    raw_snd = entropy(raw_mat.softmax(dim=-1)).mean()

    # print all
    score_others = [raw_ent.item(), raw_im.item(), raw_snd.item(), raw_softmaxCorr.item()]
    otherStr = ' '.join(['{:.4f}'.format(x) for x in score_others])

    score_ours = [target_acc, ice_intra]
    ourStr = ' '.join(['{:.4f}'.format(x) for x in score_ours])

    num_mix = [all_same_idx.shape[0], all_diff_idx.shape[0]]
    numStr = ' '.join(['{}'.format(x) for x in num_mix])
    
    log_str = "Ep:{}, synth:{}, lam:{:.4f}, score:{}, tgtAcc and ice_intra:{}, numMix:{}.\n".\
            format(args.epoch, args.synth, mix_lam, otherStr, ourStr, numStr)

    config["total_log_file"].write(log_str)
    config["task_log_file"].write(log_str)
    config["task_log_file"].write('\r\n')
    print(log_str)

def val(config, args):

    # Load models and extract parameters
    net = init_model(args)
    if args.method == 'MDD':
        net.load_state_dict(load_ckpt_mdd(args))
    else:
        net.load_state_dict(load_ckpt(args))

    mixval(net, args, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validation for unsupervised learning')

    # task parameters
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'visda', 'office-home'], help="dataset")
    parser.add_argument('--seed', type=int, default=123, help="seed")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=1, help='traversing epochs on target data')
    parser.add_argument('--run', type=int, default=0, help='run three times with three shuffled target data lists')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', '2d_uda'])

    # model parameters
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('--width', type=int, default=2048, help="for mdd")
    parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'MCC', 'BNM', 'MDD', 'ATDOC', 'SAFN', 'PADA'])
    parser.add_argument('--hyperparam', type=float, default=1.0)
    parser.add_argument('--synth', type=str, default='mixup', choices=['mixup'])
    parser.add_argument('--lam', type=float, default=0.55, help="the fixed value used in our inference-stage mixup, i.e., mixup")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    s = args.s
    t = args.t
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.s_dset_path = './data/' + args.dset + '/' + names[s] + '_list.txt'
        if args.da in {'uda', '2d_uda'}:
            args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_list_shuffle'+str(args.run)+'.txt'
        elif args.da == 'pda':
            args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_25_list_shuffle'+str(args.run)+'.txt'
        args.class_num = 65

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.s_dset_path = './data/' + args.dset + '/' + names[s] + '_list.txt'
        if args.da in {'uda', '2d_uda'}:
            args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_list_shuffle'+str(args.run)+'.txt'
        elif args.da == 'pda':
            args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_10_list_shuffle'+str(args.run)+'.txt'
        args.class_num = 31

    if args.dset == 'visda':
        names = ['training', 'validation']
        args.s_dset_path = './data/visda17/train_list.txt'
        args.t_dset_path = './data/visda17/validation_list_shuffle'+str(args.run)+'.txt'
        args.class_num = 12

    config = {}
    config['visda'] = (args.dset == 'visda')
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config['name'] = args.dset + '/' + names[s][0].upper() + names[t][0].upper()

    args.ckpt_path = os.path.join('./ckpts/'+args.da, args.method, config['name'], str(args.hyperparam)+'_final.pt')
    
    config["output_path"] = os.path.join('./logs/intra/'+args.da, args.da+'_'+str(args.epoch)+'ep_run'+str(args.run)+'_'+args.synth+\
                        '_lam'+str(args.lam), args.method, config['name'], str(args.hyperparam))

    if args.da == '2d_uda':
        args.ckpt_path = os.path.join('./ckpts/'+args.da, args.method, config['name'], str(args.hyperparam)+'_'+str(args.bottleneck_dim)+'_final.pt')
        config["output_path"] = os.path.join('./logs/intra/'+args.da, args.da+'_'+str(args.epoch)+'ep_run'+str(args.run)+'_'+args.synth+\
                            '_lam'+str(args.lam), args.method, config['name'], str(args.hyperparam)+'_'+str(args.bottleneck_dim))

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["total_log_file"] = open(osp.join('./logs/intra/'+args.da, args.da+'_'+str(args.epoch)+'ep_run'+str(args.run)+'_'+args.synth+\
                            '_lam'+str(args.lam), args.da+'_1ep_run'+str(args.run)+args.synth+'_lam'+str(args.lam)+"_log.txt"), "a+")
    config["task_log_file"] = open(osp.join(config["output_path"], str(args.hyperparam)+"_log.txt"), "a+")
    config["task_pred_file"] = osp.join(config["output_path"], str(args.hyperparam)+"_pred.pt")
    config["img_intra_list"] = osp.join(config["output_path"], str(args.hyperparam)+"_intra_cluster_list.txt")

    if args.da == '2d_uda':
        config["total_log_file"] = open(osp.join(config["output_path"], str(args.hyperparam)+'_'+str(args.bottleneck_dim)+"_log.txt"), "a+")
        config["task_pred_file"] = osp.join(config["output_path"], str(args.hyperparam)+'_'+str(args.bottleneck_dim)+"_pred.pt")
        config["img_intra_list"] = osp.join(config["output_path"], str(args.hyperparam)+'_'+str(args.bottleneck_dim)+"_intra_cluster_list.txt")


    config["prep"] = {'params':{"resize_size":256, "crop_size":224, "alexnet":False}}
    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.bs}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":args.bs}, \
                      "test_tgt":{"list_path":args.t_dset_path, "batch_size":args.bs}, \
                      "test_src":{"list_path":args.s_dset_path, "batch_size":args.bs}}

    config["total_log_file"].flush()
    config["task_log_file"].flush()

    setting_str = "dset: {}, src: {}, tgt: {}, method: {}, hyperparam: {:.4f}, net: {}.\n"\
                .format(args.dset, names[args.s], names[args.t], args.method, args.hyperparam, args.net)
    config["total_log_file"].write(setting_str)
    config["task_log_file"].write(setting_str)
    print(setting_str)

    val(config, args)
