import torch
import time
import os
import sys
import pdb

import torch
import torch.distributed as dist
import torch.nn as nn

from utils import AverageMeter, calculate_accuracy


def freeze_bn(model):
    print("Freezing Mean/Var of BatchNorm2D.")
    print("Freezing Weight/Bias of BatchNorm2D.")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False,
                rpn=None):
    print('train at epoch {}'.format(epoch))

    model.train()
    if rpn is not None:
        rpn.eval()
    else:
        freeze_bn(model)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        if rpn is not None:
            if i > 0 and N != len(inputs):
                print(inputs.shape)
                continue
            N, C, T, H, W = inputs.size()

            interval = 16
            idx = torch.arange(0,T,interval)
            rpn_inputs = inputs[:,:,idx]
            rpn_inputs = rpn_inputs.transpose(1,2).contiguous().view(N*(T//interval),C,H,W)
            try:
                with torch.no_grad():
                    proposals = rpn(rpn_inputs)
                proposals = torch.cat((proposals)).view(N,T//interval,10,4)
            except :
                pdb.set_trace()
                continue
            outputs = model(inputs, proposals.detach())
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': current_lr
            })
        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                            i + 1,
                                                            len(data_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time,
                                                            loss=losses,
                                                            acc=accuracies))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)
