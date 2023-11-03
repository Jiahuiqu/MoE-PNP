import torch
import torch.nn as nn
import random
from scipy.io import savemat
from torch import optim
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from model import Main
from dataloader import MyDataset
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import warnings
import shutil
import math
import time
import cv2
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='pavia', choices=['pavia', 'houston'])
parser.add_argument('--data', default="../dataset/pavia", metavar='DIR')
parser.add_argument('--type', default='_Elastic600')
parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('-p', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='False', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default="0", type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float, help='cross entropy loss weight')
parser.add_argument('--beta', default=0.35, type=float, help='supervised contrastive loss weight')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--seed', default=1001, type=int, help='seed for initializing training')
parser.add_argument('--store_name_define', default='reg_MS', help='保存名字的尾巴')
args = parser.parse_args()

def CC_function(ref, tar):
    # Get dimensions
    batch, bands, rows, cols = tar.shape
    tar = tar.detach().cpu().numpy()
    ref = ref.detach().cpu().numpy()
    # Initialize output array
    out = np.zeros((batch, bands))
    for b in range(batch):
        # Compute cross correlation for each band
        for i in range(bands):
            tar_tmp = tar[b, i, :, :]
            ref_tmp = ref[b, i, :, :]
            cc = np.corrcoef(tar_tmp.flatten(), ref_tmp.flatten())
            out[b, i] = cc[0, 1]

    return np.mean(out)
def cross_correlation_loss(I, J, n):
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    if args.gpu:
        sum_filter = sum_filter.cuda(args.gpu)
    I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1,1))
    J_sum = torch.conv2d(J, sum_filter,  padding=1 ,stride=(1,1))
    I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1,1))
    J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1,1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1,1))
    win_size = n**2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
    return torch.mean(cc)

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def UseDSP(HS, DSP, STF):
    DSP = DSP.permute(0, 2, 3, 1)
    HS = HS.permute(0, 2, 3, 1)
    HS_R = STF(HS, DSP).permute(0, 3, 1, 2)
    return HS_R

def UseR(HS, R):
    I = torch.mul(HS, R)
    I = torch.unsqueeze(torch.sum(I, dim=1), 1)
    return I

def gradient2d(input):
    [B, C, H, W] = input.size()
    dy = input[:, :, 1:, :] - input[:, :, :-1, :]
    dx = input[:, :, :, 1:] - input[:, :, :, :-1]
    dy = torch.concat([torch.zeros([B, C, 1, W]).to(device), dy], dim=-2)
    dx = torch.concat([torch.zeros([B, C, H, 1]).to(device), dx], dim=-1)

    return dx, dy

def hs_gradients(HS):
    dx, dy = gradient2d(HS)
    dx = torch.abs(dx)
    dy = torch.abs(dy)
    grad = torch.sum(dx+dy, dim=1)
    return grad

def dsp_smooth(dsp):
    dx, dy = gradient2d(dsp)
    dx = torch.sum(dx, dim=1)
    dy = torch.sum(dy, dim=1)
    return dx, dy

def reg_loss(inupt):
    dx, dy = gradient2d(inupt)
    dx = torch.abs(dx)
    dy = torch.abs(dy)
    grad = torch.sum(dx+dy, dim=1)
    return grad


def main(dataset=None):


    args.type = dataset

    args.store_name = '_'.join(
        [args.dataset, 'batchsize', str(args.batch_size), 'epochs', str(args.epochs),
         'lr', str(args.lr), args.store_name_define, args.type])

    if not os.path.exists(os.path.join('.', args.root_log)):
        os.mkdir(os.path.join('.', args.root_log))

    if not os.path.exists(os.path.join('.', args.root_log, args.store_name)):
        os.mkdir(os.path.join('.', args.root_log, args.store_name))


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    shutil.copyfile('./model.py', os.path.join('.', args.root_log, args.store_name, 'model.py'))

    argsDict = args.__dict__
    with open(os.path.join('.', args.root_log, args.store_name, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model")
    if args.dataset == 'pavia':
        model = Main(size_E=80, bands=102, args=args)
    elif args.dataset == '':
        pass
    else:
        raise NotImplementedError('This dataset is not supported')
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    """ optionally resume from a checkpoint """
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
                best_loss = best_loss.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    """ end """

    """ dataset """
    train_dataset = MyDataset(args.data, "train", args.type)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset = MyDataset(args.data, "test", args.type)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    """ end """

    """ tensorboard """
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    """ end """

    """ loss """
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()
    loss_NCC = NCC()
    """ end """

    best_loss = 200
    best_l1_loss = 200

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_lr(optimizer, epoch, args)

        # train for one epoch
        loss, l1_loss = train(train_loader, model, loss_l1, loss_l2, loss_NCC, optimizer, epoch, args, tf_writer)
        scheduler.step()

        # evaluate on validation set
        validate(val_loader, model, loss_l1, epoch, args, tf_writer)

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if is_best:
            best_l1_loss = l1_loss
        print('Best loss: {:.3f}, Best L1loss: {:.3f}'.format(best_loss, best_l1_loss))

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, loss_l1, loss_l2, loss_NCC, optimizer, epoch, args, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    loss_all = AverageMeter('Loss', ':.4e')
    l1_loss_all = AverageMeter('L1_Loss', ':.4e')
    reg_loss_all = AverageMeter('REG_Loss', ':.4e')
    smo_loss_all = AverageMeter('SMO_Loss', ':.4e')

    model.train()
    end = time.time()
    for step, (lrhs, pan, gths) in enumerate(train_loader):
        transform = transforms.GaussianBlur(15, 1)
        pan = pan.type(torch.float).cuda(args.gpu)
        lrhs = lrhs.type(torch.float).cuda(args.gpu)
        gths = gths.type(torch.float).cuda(args.gpu)
        gtmrhs = torch.nn.functional.interpolate(transform(gths), scale_factor=1/2, mode='bilinear', align_corners=False)
        batch_size = pan.shape[0]
        HRHS, MRHS, I_M, I_H, PAN_M, PAN_H, DSPM1, DSPH1, DSPM_T, DSPH_T, STF_M, STF_H = model(pan, lrhs)
        """LOSS区域"""
        # L1 loss
        loss1 = 1/2*loss_l1(HRHS, gths) + 1/2*loss_l1(MRHS, gtmrhs)

        # reg loss
        PAN_M_reged = UseDSP(PAN_M, DSPM1, STF_M)
        PAN_H_reged = UseDSP(PAN_H.detach(), DSPH1, STF_H)
        I_M_reged = UseDSP(I_M, DSPM_T, STF_M)
        I_H_reged = UseDSP(I_H.detach(), DSPH_T, STF_H)
        loss2_1 = loss_NCC.loss(I_M, PAN_M_reged)
        loss2_2 = loss_NCC.loss(I_H.detach(), PAN_H_reged)
        loss2_3 = loss_NCC.loss(PAN_M, I_M_reged)
        loss2_4 = loss_NCC.loss(PAN_H.detach(), I_H_reged)
        loss2 = (loss2_1 + loss2_2 + loss2_3 + loss2_4) / 4

        # smooth loss
        loss3_1 = smooothing_loss(DSPM1)
        loss3_2 = smooothing_loss(DSPH1)
        loss3_3 = smooothing_loss(DSPM_T)
        loss3_4 = smooothing_loss(DSPH_T)
        loss3 = (loss3_1 + loss3_2 + loss3_3 + loss3_4)/4


        loss = loss1 + 0.001*loss2 + 0.01*loss3

        # loss updata
        loss_all.update(loss.item(), batch_size)
        l1_loss_all.update(loss1.item(), batch_size)
        reg_loss_all.update(loss2.item(), batch_size)
        smo_loss_all.update(loss3.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'L1 Loss {l1_loss.val:.4f}\t'
                      'REG Loss {reg_loss.val:.4f}\t'
                      'SMO Loss {smo_loss.val:.4f}'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                loss=loss_all, l1_loss=l1_loss_all, reg_loss=reg_loss_all, smo_loss=smo_loss_all))
            print(output)

    # tensorboard
    tf_writer.add_scalar('Loss/train', loss_all.avg, epoch)
    tf_writer.add_scalar('L1 loss/train', l1_loss_all.avg, epoch)
    tf_writer.add_scalar('REG loss/train', reg_loss_all.avg, epoch)
    tf_writer.add_scalar('SMO loss/train', smo_loss_all.avg, epoch)
    tf_writer.add_image("P_reged", PAN_M_reged[0,:,:,:].cpu().detach().numpy(), epoch, dataformats='CHW')
    tf_writer.add_image("I_reged", I_M_reged[0, :, :, :].cpu().detach().numpy(), epoch, dataformats='CHW')
    tf_writer.add_image("reg_I", PAN_M[0,:,:,:].cpu().detach().numpy(), epoch, dataformats='CHW')
    tf_writer.add_image("DSP", show_flow_hsv(DSPM1.cpu().detach().numpy()), epoch, dataformats='HWC')
    tf_writer.add_image("gths", I_M[0, 0, :, :].cpu().detach().numpy(), epoch, dataformats='HW')

    return loss_all.avg, l1_loss_all.avg

def validate(val_loader, model, loss_l1, epoch, args, tf_writer):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    l1_loss_all = AverageMeter('L1_Loss', ':.4e')
    CC = AverageMeter('CC', ':6.2f')

    with torch.no_grad():
        end = time.time()
        for step, (lrhs, pan, gths) in enumerate(val_loader):
            pan = pan.type(torch.float).cuda(args.gpu)
            lrhs = lrhs.type(torch.float).cuda(args.gpu)
            gths = gths.type(torch.float).cuda(args.gpu)
            batch_size = pan.shape[0]
            HRHS, MRHS, I_M, I_H, PAN_M, PAN_H, DSPM1, DSPH1, DSPM_T, DSPH_T, _, __ = model(pan, lrhs)

            # L1 loss
            loss1 = loss_l1(HRHS, gths)

            CC_val = CC_function(HRHS, gths)
            l1_loss_all.update(loss1.item(), batch_size)
            CC.update(CC_val, batch_size)

            batch_time.update(time.time() - end)

        output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'l1_loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})\t'
                  'CC {CC.val:.3f} ({CC.avg:.3f})'.format(
            epoch, len(val_loader), batch_time=batch_time, l1_loss=l1_loss_all, CC=CC, ))
        print(output)

        tf_writer.add_scalar('L1 loss/val', l1_loss_all.avg, epoch)
        tf_writer.add_scalar('CC/val', CC.avg, epoch)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.root_log, args.store_name, 'MDF_NREG.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    if state['epoch']%10==0:
        filename = os.path.join(args.root_log, args.store_name, 'MDF_NREG_'+str(state['epoch'])+'.pth.tar')
        torch.save(state, filename)


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def show_flow_hsv(flow, show_style=2):
    flow = np.transpose(flow[0, :, :, :], (1, 2, 0))
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # 将直角坐标系光流场转成极坐标系

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2 #angle弧度转角度
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # magnitude归到0～255之间
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

# hsv to bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(torch.nn.functional, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
if __name__ == "__main__":
    # main(dataset='_Elastic3000')
    # main(dataset='_Elastic600')
    # main(dataset='_Elastic500')
    # main(dataset='_Elastic400')
    # main(dataset='_Elastic300')
    # main(dataset='_Elastic200')
    # main(dataset='_Elastic100')
    main(dataset='')
