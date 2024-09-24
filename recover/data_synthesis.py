'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import argparse
import collections
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from utils import *
from torch.utils.data import Dataset

def embed(model, input):
    #fc = list(model.children())[-1]
    embed  = nn.Sequential(*list(model.children())[:-1]).append(nn.Flatten())
    embeding = embed(input)
    #print(embeding.shape)
    #print(fc)
    #print()
    #print(embed)
    return embeding#,fc(embeding)
    

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    
    
def get_images(args, model_teacher, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 100
    #batch_size = args.batch_size

    best_cost = 1e4

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    # setup target labels
    # targets_all = torch.LongTensor(np.random.permutation(1000))
    # targets_all = torch.LongTensor(np.arange(1000))

    #load data real and syn

    #real
    channel = 3
    im_size = (64, 64)
    num_classes = 200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data = torch.load(os.path.join('../', 'tinyimagenet.pt'), map_location='cpu')

    class_names = data['classes']

    images_train = data['images_train']
    labels_train = data['labels_train']
    images_train = images_train.detach().float() / 255.0
    labels_train = labels_train.detach()
    for c in range(channel):
        images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
    dst_train = TensorDataset(images_train, labels_train)  # no augmentation
    args.device = torch.device('cuda')

    # 不用val
    # images_val = data['images_val']
    # labels_val = data['labels_val']
    # images_val = images_val.detach().float() / 255.0
    # labels_val = labels_val.detach()

    # for c in range(channel):
    #     images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

    # dst_test = TensorDataset(images_val, labels_val)  # no augmentation
    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
    args.ipc = 50
    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    real_batch = 200
    #去掉循环
    #for kk in range(0, 1000, batch_size):
    #targets = targets_all[kk:min(kk+batch_size,1000)].to('cuda')

    data_type = torch.float
    # inputs = torch.randn((targets.shape[0], 3, 224, 224), requires_grad=True, device='cuda',
    #                      dtype=data_type)

    iterations_per_layer = args.iteration
    lim_0, lim_1 = args.jitter , args.jitter

    optimizer = optim.Adam([image_syn], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
    lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer) # 0 - do not use warmup
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    for iteration in range(iterations_per_layer):
        # learning rate scheduling
        lr_scheduler(optimizer, iteration, iteration)

        aug_function = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        loss_fm,loss_bn = torch.tensor(0.0).cuda(),torch.tensor(0.0).cuda()

        for c in range(num_classes):
            idx_shuffle = np.random.permutation(indices_class[c])[:real_batch]
            img_real = images_all[idx_shuffle]

            img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

            syn_jit = aug_function(img_syn)
            real_jit = aug_function(img_real)

            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            syn_jit = torch.roll(syn_jit, shifts=(off1, off2), dims=(2, 3))
            real_jit = torch.roll(real_jit, shifts=(off1, off2), dims=(2, 3))
            # forward pass
            #optimizer.zero_grad()
            #不知道embed和fc是否可用
            real_embed = embed(model_teacher,real_jit)

            real_bn_sta = []
            for bn in loss_r_feature_layers:
                real_bn_sta.append((bn.mean,bn.var))

            syn_embed = model_teacher.embed(syn_jit)
            #syn_outputs = model_teacher.fc(syn_embed)

            syn_bn_sta = []
            for bn in loss_r_feature_layers:
                syn_bn_sta.append((bn.mean,bn.var))

            loss_bn_temp = [(torch.norm(real_mean - syn_bn_sta[idx][0], 2) + torch.norm(real_var - syn_bn_sta[idx][1], 2)) for (idx,(real_mean,real_var)) in enumerate(real_bn_sta)]
            rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
            loss_bn += sum([loss * rescale[idx] for (idx, loss) in enumerate(loss_bn_temp)])

            #loss_ce += criterion(syn_outputs,torch.full(syn_outputs.shape[0],c))
            loss_fm += torch.sum((torch.mean(real_embed, dim=0) - torch.mean(syn_embed, dim=0))**2)
            
        

        # # R_cross classification loss
        # loss_ce = criterion(outputs, targets)

        # # R_feature loss
        # rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
        # loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

        # # R_prior losses
        # _, loss_var_l2 = get_image_prior_losses(inputs_jit)

        # # l2 loss on images
        # loss_l2 = torch.norm(inputs_jit.reshape(batch_size, -1), dim=1).mean()

        # # combining losses
        # loss_aux = args.tv_l2 * loss_var_l2 + \
        #             args.l2_scale * loss_l2 + \
        #             args.r_bn * loss_r_bn_feature

        loss =   loss_bn + loss_fm  #+loss_ce

        if iteration % save_every==0:
            print("------------iteration {}----------".format(iteration))
            print("total loss", loss.item())
            print("loss_r_bn_feature", loss_bn.item())
            print("feature loss", loss_fm.item())
            # comment below line can speed up the training (no validation process)
            # if hook_for_display is not None:
            #     hook_for_display(inputs, targets)

        # do image update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clip color outlayers
        image_syn.data = clip(image_syn.data)

        # if best_cost > loss.item() or iteration == 1:    #并没有更新best_cost，而且iteration也是从0开始的
        #     best_inputs = inputs.data.clone()

    if args.store_best_images:
        best_inputs = image_syn.data.clone() # using multicrop, save the last one
        best_inputs = denormalize(best_inputs)
        save_images(args, best_inputs, label_syn, ipc_id)

    # to reduce memory consumption by states of the optimizer we deallocate memory
    optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()

def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path +'/class{:03d}_id{:03d}.jpg'.format(class_id,ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def parse_args():
    parser = argparse.ArgumentParser(
        "SRe2L: recover data from pre-trained model")
    """Data save flags"""
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--tv-l2', type=float, default=0.0001,
                        help='coefficient for total variation L2 loss')
    parser.add_argument('--l2-scale', type=float,
                        default=0.00001, help='l2 loss on the image')
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    """IPC (Image Per Class) flags"""
    parser.add_argument("--ipc-start", default=0, type=int, help="start index of IPC")
    parser.add_argument("--ipc-end", default=50, type=int, help="end index of IPC")
    args = parser.parse_args()

    args.syn_data_path= os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    return args


def main_syn(ipc_id):
    model_teacher = models.__dict__[args.arch_name](pretrained=True)
    model_teacher = model_teacher.cuda()
    #print('test')
    #fe,logits = embed(model_teacher,torch.randn(size=(2, 3, 64, 64), dtype=torch.float, requires_grad=True, device=torch.device('cuda')))
    #print(fe.shape,fe)
    #print(logits.shape,logits)
    #print('finish')
    #exit()
    #gai
    #model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    # model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
    # model_verifier = model_verifier.cuda()
    # model_verifier.eval()
    # for p in model_verifier.parameters():
    #     p.requires_grad = False

    hook_for_display = None#lambda x,y: validate(x, y, model_verifier)
    get_images(args, model_teacher, hook_for_display, ipc_id)


if __name__ == '__main__':
    args = parse_args()
    # for ipc_id in range(0,50):
    # for ipc_id in range(args.ipc_start, args.ipc_end):
    #     print('ipc = ', ipc_id)
    #     main_syn(ipc_id)
    #修改
    main_syn(0)
