# -*- coding:utf-8 -*-
from time import time
import os
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
sys.path.append('./loss')
sys.path.append('./net')
sys.path.append('./dataset')
from Dice_loss import DiceLoss
from VNet_kernel3_dropout import net
from stage1_dataset import train_fix_ds, val_fix_ds


LOG_DIR = './log.txt'
log = open(LOG_DIR,'w')

def print_out(log,txt):
    log.write(txt+'\n')
    log.flush()
    print(txt)

# 定义超参数
on_server = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1,2'
cudnn.benchmark = True
Epoch = 100
leaing_rate_base = 1e-4
module_dir = './save_module/net70-0.000.pth'
pre_train = False
train_batch_size = 1 if on_server is False else 4
val_batch_size = 2
num_workers = 1 if on_server is False else 2
pin_memory = False if on_server is False else True

net = torch.nn.DataParallel(net).cuda()
if pre_train:
    net.load_state_dict(torch.load(module_dir))
    net.eval()
# 定义数据加载
train_dl = DataLoader(train_fix_ds, train_batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
val_dl = DataLoader(val_fix_ds, val_batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
# 定义损失函数
loss_func = DiceLoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [20, 40, 60])

model_path = './module'
if not os.path.exists(model_path): os.makedirs(model_path)

def plot_loss(trainLoss, valLoss):
    x = range(trainLoss)
    plt.plot(x, trainLoss, color='blue', label='train loss')
    plt.plot(x, valLoss, color='red', label = 'val loss')
    plt.legend()
    plt.savefig('loss.png')
    
def val_one_epoch(net,epoch,step):
    total_loss = 0.0
    for n, (ct, seg) in enumerate(val_dl):
        ct = ct.cuda()
        seg = seg.cuda()

        # 如果一个正样本都没有就直接结束
        if torch.numel(seg[seg > 0]) == 0:
            continue

        ct = Variable(ct)
        seg = Variable(seg)

        outputs = net(ct)
        loss = loss_func(outputs, seg)
        total_loss += loss.item()
    print_out(log, 'epoch:{}, step:{}, val loss:{:.3f}, time:{:.3f} min'.format(epoch,step,total_loss/(n+1),(time() - start) / 60))
    return total_loss/(n+1)
# 训练网络
train_loss = []
val_loss = []
start = time()
for epoch in range(Epoch):

    lr_decay.step()

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        # 如果一个正样本都没有就直接结束
        if torch.numel(seg[seg > 0]) == 0:
            continue

        ct = Variable(ct)
        seg = Variable(seg)

        outputs = net(ct)
        loss = loss_func(outputs, seg)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 is 0:
            print_out(log,'epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.item(), (time() - start) / 60))
            #train_loss.append(loss.item())
            #val_loss_value = val_one_epoch(net, epoch, step)
            #val_loss.append(val_loss_value)
            #plot_loss(train_loss, val_loss)
    # 每十个个epoch保存一次模型参数
    if (epoch+1) % 5 is 0:
        torch.save(net.state_dict(), os.path.join(model_path,'net{}-{:.3f}.pth'.format(epoch, loss.item())))
    
log.close()



    
