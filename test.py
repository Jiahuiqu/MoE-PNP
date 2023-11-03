import torch as t
import torch.nn as nn
from scipy.io import savemat
from torch.utils.data import DataLoader
from model import Main
from dataloader import MyDataset
import argparse

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

# 超参数

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default="0", type=int,
                    help='GPU id to use.')

def test(root, type):
    args = parser.parse_args()
    # 数据准备
    data = MyDataset(root, "test", type)
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    # 模型加载
    model = Main(80, 102, args)
    model = model.cuda(args.gpu)
    model.load_state_dict(t.load('./log/pavia_batchsize_4_epochs_80_lr_0.0001_reg_MS_/MDF_NREG_80.pth.tar')['state_dict'])
    model.eval()
    loss_fun = nn.L1Loss()

    # 模型测试
    for step, (lrhs, pan, gths) in enumerate(data_loader):
        pan = pan.type(t.float).cuda(args.gpu)
        lrhs = lrhs.type(t.float).cuda(args.gpu)
        gths = gths.type(t.float).cuda(args.gpu)

        HRHS, MRHS, HS_M_loss, HS_H_loss, PAN_M, PAN_H, DSPM1, DSPH1, _, _, _, _ = model(pan, lrhs)
        loss = loss_fun(HRHS, gths)

        print('step:' + str(step), '--loss:' + str(loss.data))
        savemat('./dataout/pavia' + type + '/'+str(step+1)+".mat", {'out': HRHS.cpu().detach().numpy()})


if __name__ == "__main__":
    # setup_seed(67)
    root = "../dataset/pavia"
    # test(root, '_Elastic100')
    test(root, '')
    # test(root, '_alpha1')
