import time
from torch import nn
import ops.meters as meters
import ops.tests as tests
import torch
import torchmetrics

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import numpy as np


def train(model, train_iter, valid_iter, optimizer, train_schedule, warmup_scheduler, device, writer=None, verbose=1):

    print('training on: ', device)
    model.to(device)
    warmup_time = time.time()
    # 热身阶段
    warmup_epochs = 10
    for epoch in range(warmup_epochs):
        batch_time = time.time()
        *metrics, = train_epoch(model, train_iter, optimizer, warmup_scheduler, device)
        batch_time = time.time() - batch_time
        template = '(%.2f sec/epoch) Warmup Epoch: %d, Loss: %.4f, Acc: %.4f, Precision: %.4f, Recall: %.4f, F1_score: %.4f, lr: %.3e'
        print(template % (batch_time, epoch, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], [param_group['lr'] for param_group in optimizer.param_groups][0]))

        if writer is not None and (epoch + 1) % 1 == 0:
            *test_metrics, = tests.valid(model, data_iter=valid_iter, verbose=True, device=device)

    if warmup_epochs > 0:
        print('The model is warmed up: %.2f sec\n' % (time.time() - warmup_time))

    # 正式训练阶段
    epochs = 200
    best_valid_acc = 0
    for epoch in range(epochs):
        batch_time = time.time()
        *metrics, = train_epoch(model, train_iter, optimizer, train_schedule, device)
        batch_time = time.time() - batch_time

        if writer is not None and (epoch + 1) % 1 == 0:
            add_train_metrics(writer, metrics, epoch, [param_group['lr'] for param_group in optimizer.param_groups][0])
            template = '(%.2f sec/epoch) Epoch: %d, Loss: %.4f, Acc: %.4f, Precision: %.4f, Recall: %.4f, F1_score: %.4f, lr: %.3e'
            print(template % (batch_time, epoch, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], [param_group['lr'] for param_group in optimizer.param_groups][0]))
        if writer is not None and (epoch + 1) % 1 == 0:
            *test_metrics, = tests.valid(model, data_iter=valid_iter, verbose=True, device=device)
            add_valid_metrics(writer, test_metrics, epoch)

            if verbose > 1:
                for name, param in model.named_parameters():
                    name = name.split(".")
                    writer.add_histogram("%s/%s" % (name[0], ".".join(name[1:])), param, global_step=epoch)

        if best_valid_acc < test_metrics[1]:
            best_valid_acc = test_metrics[1]
            torch.save(model, './save_weight/reflect_model.pkl')


def train_epoch(model, data_iter, optimizer, scheduler, device):
    
    model.to(device)
    model.train()
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    loss_meter = meters.AverageMeter('loss')
    acc_meter = meters.AverageMeter('acc')
    precision_meter = meters.AverageMeter('precision')
    recall_meter = meters.AverageMeter('recall')
    f1_meter = meters.AverageMeter('f1_score')

    for step, (xs, ys) in enumerate(data_iter):
        xs = xs.to(device)
        ys = ys.to(device)

        optimizer.zero_grad()
        logits = model(xs)
        loss = loss_function(logits, ys)
        *total_metrics, = classification_metrics(logits, ys)
        loss_meter.update(loss.item())
        acc_meter.update(total_metrics[0].item())
        precision_meter.update(total_metrics[1].item())
        recall_meter.update(total_metrics[2].item())
        f1_meter.update(total_metrics[3].item())
        loss.backward()
        # 梯度裁剪
        '''
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        '''
        optimizer.step()

        if scheduler:
            scheduler.step()

    return loss_meter.avg, acc_meter.avg, precision_meter.avg, recall_meter.avg, f1_meter.avg


def classification_metrics(logits, ys):
    # logit_amax = np.amax(logits.cpu().detach().numpy(), axis=1)
    logits_argmax = logits.argmax(dim=-1).to('cpu')
    ys = ys.to('cpu')
    accuracy = accuracy_score(logits_argmax, ys, )
    precision = precision_score(logits_argmax, ys, average='macro', zero_division=1)
    recall = recall_score(logits_argmax, ys, average='macro', zero_division=1)
    f1 = f1_score(logits_argmax, ys, average='macro', zero_division=1)

    return accuracy, precision, recall, f1


def add_train_metrics(writer, metrics, epoch, lr):

    loss, acc, precision, recall, f1_score = metrics

    writer.add_scalar("train/loss", loss, global_step=epoch)
    writer.add_scalar("train/acc", acc, global_step=epoch)
    writer.add_scalar("train/precision", precision, global_step=epoch)
    writer.add_scalar("train/recall", recall, global_step=epoch)
    writer.add_scalar("train/f1_score", f1_score, global_step=epoch)
    writer.add_scalar("train/learning-rate", lr, global_step=epoch)


def add_valid_metrics(writer, metrics, epoch):
    loss, acc, precision, recall, f1_score = metrics

    writer.add_scalar("valid/loss", loss, global_step=epoch)
    writer.add_scalar("valid/acc", acc, global_step=epoch)
    writer.add_scalar("valid/precision", precision, global_step=epoch)
    writer.add_scalar("valid/recall", recall, global_step=epoch)
    writer.add_scalar("valid/f1_score", f1_score, global_step=epoch)


