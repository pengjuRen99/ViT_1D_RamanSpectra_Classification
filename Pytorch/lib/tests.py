import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import joblib

import ops.meters as meters
import ops.trains as trains
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score,f1_score,recall_score


@torch.no_grad()
def valid(model, data_iter, verbose=False, period=10, device=None, ):
    
    model.to(device)
    model.eval()

    loss_meter = meters.AverageMeter('loss')
    acc_meter = meters.AverageMeter('acc')
    precision_meter = meters.AverageMeter('precision')
    recall_meter = meters.AverageMeter('recall')
    f1_meter = meters.AverageMeter('f1_score')


    for step, (xs, ys) in enumerate(data_iter):

        loss_function = nn.CrossEntropyLoss()

        xs = xs.to(device)
        ys_pred = model(xs).cpu()

        loss_meter.update(loss_function(ys_pred, ys).numpy())

        # 计算准确率
        *test_metrics, = trains.classification_metrics(ys_pred, ys)
        acc_meter.update(test_metrics[0].item())
        precision_meter.update(test_metrics[1].item())
        recall_meter.update(test_metrics[2].item())
        f1_meter.update(test_metrics[3].item())

        loss_value = loss_meter.avg
        acc_value = acc_meter.avg
        precision_value = precision_meter.avg
        recall_value = recall_meter.avg
        f1_value = f1_meter.avg

        metrics = loss_value, acc_value, precision_value, recall_value, f1_value,
        if verbose and int(step + 1) % period == 0:
            print("%d Steps, %s" % (int(step + 1), repr_metrics(metrics)))
        
    print(repr_metrics(metrics))

    return (*metrics,)


@torch.no_grad()
def test(model, data_iter, device, ):
    model.to(device)
    model.eval()

    y_pred = np.array([])
    y_true = np.array([])

    for step, (xs, ys) in enumerate(data_iter):

        xs = xs.to(device)
        ys_pred = model(xs).cpu()

        y_pred = np.append(y_pred, ys_pred.argmax(dim=-1).detach().numpy())
        y_true = np.append(y_true, ys.cpu().numpy())

    test_metrics_display(y_true, y_pred)


def repr_metrics(metrics):
    loss_value, acc_value, precision_value, recall_value, f1_value = metrics

    metrics_reprs = [
        'Valid Dataset: ',
        'Loss: %.4f' % loss_value,
        'Acc: %.4f' % acc_value,
        'Precision: %.4f' % precision_value,
        'Recall: %.4f' % recall_value,
        'F1: %.4f' % f1_value,
    ]

    return ', '.join(metrics_reprs)


def test_metrics_display(y_true, y_pred):
    print('Classification_report', classification_report(y_true, y_pred, digits=4, zero_division=1))
    print('accuracy_score', accuracy_score(y_true, y_pred))
    print('------Weighted------')
    print('Weighted precision', precision_score(y_true, y_pred, average='weighted', zero_division=1))
    print('Weighted recall', recall_score(y_true, y_pred, average='weighted', zero_division=1))
    print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted', zero_division=1))
    plot_cm(y_true, y_pred)


def plot_cm(labels, predictions):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    cm = confusion_matrix(labels, predictions, )
    class_label = joblib.load('class_label._reflect.pkl')
    label = [label for label in class_label]
    confusion_pd = pd.DataFrame(cm, index=label, columns=label)
    f = plt.figure(figsize=(16, 10), dpi=900)
    ax = plt.subplot(111)
    sns.heatmap(confusion_pd, cmap='Blues', annot=True, fmt='g')
    plt.title('Reflection spectrum dataset confusion matrix')
    # plt.title('Transmission spectrum dataset confusion matrix')
    ax.set_ylabel('Actual label')
    ax.set_xlabel('Predicted label')
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical', )
    ax.tick_params(axis='y', )
    ax.tick_params(axis='x', labelrotation=45)
    plt.tight_layout()
    plt.savefig('resource/Reflection_confusionMatric.jpg')
    # plt.savefig('resource//Transmission_confusionMatric.jpg')
    plt.show()
