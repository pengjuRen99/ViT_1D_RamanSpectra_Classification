import tensorflow as tf
import os
import datetime
from ops.utils import generate_ds

from models.Vit import vit_base
import math
from tqdm import tqdm
import re
import sys
import os

import sklearn.metrics as sklearn_metrics
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, precision_score,f1_score,recall_score, ConfusionMatrixDisplay, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt

import shutil
from ops.split_data import SplitDataset

import seaborn as sns
import pandas as pd


def plot_metrics(epochs, history):
    # Draw loss, AUC, precision and recall curve
    plt.figure(figsize=(8, 4), dpi=900)
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric
        plt.subplot(2,2,n+1)
        plt.plot(list(range(1, epochs+1)),  history['train_'+metric], label='Train')
        plt.plot(list(range(1, epochs+1)), history['val_'+metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()
        plt.tight_layout()

def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions, )
    # Reflective dataset classes name
    # classes = ['Chinese holstein cattle', 'Red-crowned crane', 'Yunnan donkey', 'Limousin', 'Nanyang cattle', \
    # 'Charolais', 'Cat', 'Angus cattle', 'Yanbian yellow cattle', 'Xuzhou cattle', 'German yellow cattle', \
    # 'Dezhou donkey', 'Murrah buffalo', 'Shiba Inu', 'Sika deer', 'Buffalo', 'Bohai black cattle', 'Pandas', \
    # 'Dzo', 'Pig', 'Chinese pastoral dog', 'Shorthorn', 'Tengchong horse', 'British shorthair', 'Holstein cow', \
    # 'Mongolia cattle', 'Kangaroo', 'Tibet cattle', 'Tibet horse', 'Tibet donkey', 'Simmental cattle', \
    # 'Golden monkey', 'Qinghai yellow cattle', 'horse', 'Chicken', 'Duck', 'Pigeon', 'Goose', 'Yellow cattle', 'Black bear']
    # Transmissive dataset classes name 
    classes = ['Chinese holstein cattle', 'Yunnan donkey', 'Limousin', 'Nanyang cattle', 'Jungle fowl', \
    'Charolais', 'Angus cattle', 'Pheasant', 'Yanbian yellow cattle', 'Xuzhou cattle', 'German yellow cattle', \
    'Dezhou donkey', 'Murrah buffalo', 'Bohai black cattle', 'Dzo', 'Pig', \
    'Brown swiss', 'Syrmaticus reevesii', 'White-naped crane', 'White stork', 'Shorthorn', 'Golden pheasant', \
    'Tengchong horse', 'Holstein cow', 'Mongolia cattle', 'Demoiselle crane', 'Blue-and-yellow Macaw', \
    'Tibet cattle', 'Tibet horse', 'Tibet donkey', 'Simmental cattle', 'Liao white cow', 'Golden monkey', \
    'Leopard', 'Gibbon', 'Qinghai yellow cattle', 'horse', 'Chicken', 'Duck', 'Pigeon', 'Goose', 'Black bear']
    # sns.heatmap(cm, cmap='Oranges')
    confusion_pd = pd.DataFrame(cm, index=classes, columns=classes)
    f = plt.figure(figsize=(16, 10), dpi=900)
    ax = plt.subplot(111)
    sns.heatmap(confusion_pd, cmap='Blues', annot=True, fmt='g')
    # plt.title('Reflection spectrum dataset confusion matrix')
    plt.title('Transmission spectrum dataset confusion matrix')
    ax.set_ylabel('Actual label')
    ax.set_xlabel('Predicted label')
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical', )
    ax.tick_params(axis='y', )
    ax.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    # plt.savefig('curve_fig/animal_blood/Reflection_confusionMatric.jpg')
    plt.savefig('curve_fig/animal_blood/Transmission_confusionMatric.jpg')
    plt.show()

def plot_roc(y_true, y_pred, num_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = sklearn_metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Reflection class
    # classes = ['Chinese holstein cattle', 'Red-crowned crane', 'Yunnan donkey', 'Limousin', 'Nanyang cattle', \
    # 'Charolais', 'Cat', 'Angus cattle', 'Yanbian yellow cattle', 'Xuzhou cattle', 'German yellow cattle', \
    # 'Dezhou donkey', 'Murrah buffalo', 'Shiba Inu', 'Sika deer', 'Buffalo', 'Bohai black cattle', 'Pandas', \
    # 'Dzo', 'Pig', 'Chinese pastoral dog', 'Shorthorn', 'Tengchong horse', 'British shorthair', 'Holstein cow', \
    # 'Mongolia cattle', 'Kangaroo', 'Tibet cattle', 'Tibet horse', 'Tibet donkey', 'Simmental cattle', \
    # 'Golden monkey', 'Qinghai yellow cattle', 'horse', 'Chicken', 'Duck', 'Pigeon', 'Goose', 'Yellow cattle', 'Black bear']
    # Transmission class
    classes = ['Chinese holstein cattle', 'Yunnan donkey', 'Limousin', 'Nanyang cattle', 'Jungle fowl', \
    'Charolais', 'Angus cattle', 'Pheasant', 'Yanbian yellow cattle', 'Xuzhou cattle', 'German yellow cattle', \
    'Dezhou donkey', 'Murrah buffalo', 'Bohai black cattle', 'Dzo', 'Pig', \
    'Brown swiss', 'Syrmaticus reevesii', 'White-naped crane', 'White stork', 'Shorthorn', 'Golden pheasant', \
    'Tengchong horse', 'Holstein cow', 'Mongolia cattle', 'Demoiselle crane', 'Blue-and-yellow Macaw', \
    'Tibet cattle', 'Tibet horse', 'Tibet donkey', 'Simmental cattle', 'Liao white cow', 'Golden monkey', \
    'Leopard', 'Gibbon', 'Qinghai yellow cattle', 'horse', 'Chicken', 'Duck', 'Pigeon', 'Goose', 'Black bear']

    plt.figure(figsize=(16, 10), dpi=900)
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], lw=2, label = classes[i] + '(AUC = {0:0.4f})'.format(roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Reflection spectrum of Receiver operating characteristic to multi-class')
    plt.title('Transmission spectrum of Receiver operating characteristic to multi-class')
    plt.legend(loc='lower right', ncol=2, shadow=True, fancybox=True)
    plt.tight_layout()
    plt.savefig('curve_fig/animal_blood/Transmission_ROC.jpg')
    # plt.savefig('curve_fig/animal_blood/Reflection_ROC.jpg')
    plt.show()

def plot_prc(y_true, y_pred, num_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision['micro'], recall['micro'], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    average_precision['micro'] = average_precision_score(y_true, y_pred, average='micro')
    plt.figure(figsize=(16, 10), dpi=900)
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score of transmission spectrum, micro-averaged over all classes: AP={0:0.4f}'.format(average_precision['micro']))
    # plt.title('Average precision score of reflection spectrum, micro-averaged over all classes: AP={0:0.4f}'.format(average_precision['micro']))
    plt.tight_layout()
    plt.savefig('curve_fig/animal_blood/Transmission_PRC.jpg')
    # plt.savefig('curve_fig/animal_blood/Reflection_PRC.jpg')
    plt.show()

def main():

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)
    #         exit(-1)

    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices(physical_devices[1:], 'GPU')

    shutil.rmtree('./Datasets/animal_blood/Transmissive_animalBlood_preprocessing/test')
    os.mkdir('./Datasets/animal_blood/Transmissive_animalBlood_preprocessing/test')
    shutil.rmtree('./Datasets/animal_blood/Transmissive_animalBlood_preprocessing/train')
    os.mkdir('./Datasets/animal_blood/Transmissive_animalBlood_preprocessing/train')
    shutil.rmtree('./Datasets/animal_blood/Transmissive_animalBlood_preprocessing/valid')
    os.mkdir('./Datasets/animal_blood/Transmissive_animalBlood_preprocessing/valid')

    split_dataset = SplitDataset(dataset_dir='../Data_set/Transmissive_animalBlood_preprocessing', saved_dataset_dir='Datasets/animal_blood/Transmissive_animalBlood_preprocessing', show_progress=True)
    split_dataset.start_splitting()

    # data_root = 'Datasets/animal_blood/Reflective_animalBlood_preprocessing'
    data_root = 'Datasets/animal_blood/Transmissive_animalBlood_preprocessing'

    if not os.path.exists('./save_weights'):
        os.mkdir('./save_weights')

    batch_size = 64
    epochs = 400
    # num_classes = 40            # Reflective_animalBlood_preprocessing num_classes
    num_classes = 42              # Transmissive_animalBlood_preprocessing num_classes
    initial_lr = 0.001
    weight_decay = 1e-4

    wait = 0                # init best acc account 
    patience = 30           # eraly stopping epoch

    log_dir = "./logs_animalBlood/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator 
    train_ds, val_ds, test_ds, lb = generate_ds(dataset_dir=data_root, batch_size=batch_size)

    # create model
    model = vit_base(num_classes=num_classes, has_logits=True)
    # model.build((batch_size, 1400, 1))        # Reflective_processed_data
    model.build((batch_size, 960, 1))           # Transmissive_processed data
    model.summary()

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    train_precision = tf.keras.metrics.Precision(name='precision')
    train_recall = tf.keras.metrics.Recall(name='recall')
    train_auc = tf.keras.metrics.AUC(name='auc')
    train_prc = tf.keras.metrics.AUC(name='prc', curve='PR')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
    val_precision = tf.keras.metrics.Precision(name='precision')
    val_recall = tf.keras.metrics.Recall(name='recall')
    val_auc = tf.keras.metrics.AUC(name='auc')
    val_prc = tf.keras.metrics.AUC(name='prc', curve='PR')
    

    @tf.function
    def train_step(train_spectral, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_spectral, training=True)
            # cross entropy loss
            ce_loss = loss_object(train_labels, output)

            # l2 loss
            matcher = re.compile(".*(bias|gamma|beta).*")
            l2loss = weight_decay * tf.add_n([
                tf.nn.l2_loss(v)
                for v in model.trainable_variables
                if not matcher.match(v.name)
            ])

            loss = ce_loss + l2loss
            # loss = ce_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(ce_loss)
        train_accuracy(train_labels, output)
        train_precision(train_labels, output)
        train_recall(train_labels, output)
        train_auc(train_labels, output)
        train_prc(train_labels, output)

    @tf.function
    def val_step(val_spectral, val_labels):
        output = model(val_spectral, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)
        val_precision(val_labels, output)
        val_recall(val_labels, output)
        val_auc(val_labels, output)
        val_prc(val_labels, output)

    best_val_acc = 0.
    t_loss, t_auc, t_precision, t_recall = [], [], [], []
    v_loss, v_auc, v_precision, v_recall = [], [], [], [] 
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        train_precision.reset_states()
        train_recall.reset_states()
        train_auc.reset_states()
        train_prc.reset_states()
        val_precision.reset_states()
        val_recall.reset_states()
        val_auc.reset_states()
        val_prc.reset_states()

        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for spectrals, labels in train_bar:
            train_step(spectrals, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}, precision:{:.3f}, recall:{:.3f}, auc:{:.3f}, prc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result(),
                                                                                 train_precision.result(),
                                                                                 train_recall.result(),
                                                                                 train_auc.result(),
                                                                                 train_prc.result()
                                                                                 )
        t_loss.append(train_loss.result())
        t_auc.append(train_auc.result())
        t_precision.append(train_precision.result())
        t_recall.append(train_recall.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for spectrals, labels in val_bar:
            val_step(spectrals, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}, precision:{:.3f}, recall:{:.3f}, auc:{:.3f}, prc:{:.3f}".format(epoch + 1,
                                                                                epochs,
                                                                                val_loss.result(),
                                                                                val_accuracy.result(),
                                                                                val_precision.result(),
                                                                                val_recall.result(),
                                                                                val_auc.result(),
                                                                                val_prc.result()
                                                                                )
        v_loss.append(val_loss.result())
        v_auc.append(val_auc.result())
        v_precision.append(val_precision.result())
        v_recall.append(val_recall.result())


        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)
            tf.summary.scalar("precision", train_precision.result(), epoch)
            tf.summary.scalar("recall", train_recall.result(), epoch)
            tf.summary.scalar("auc", train_auc.result(), epoch)
            tf.summary.scalar("prc", train_prc.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)
            tf.summary.scalar("precision", val_precision.result(), epoch)
            tf.summary.scalar("recall", val_recall.result(), epoch)
            tf.summary.scalar("auc", val_auc.result(), epoch)
            tf.summary.scalar("prc", val_prc.result(), epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/model.ckpt"
            model.save_weights(save_name, save_format="tf")
            wait = 0
        else:
            wait +=1 
        
        """ if wait > patience:
            print('Epoch %d: early stopping!' % (epoch+1))
            h_epoch = epoch+1
            break """
        
    history = {'train_loss': t_loss, 'train_auc': t_auc, 'train_precision': t_precision, 'train_recall': t_recall, 
                   'val_loss': v_loss, 'val_auc': v_auc, 'val_precision': v_precision, 'val_recall': v_recall
        }
    # plot_metrics(h_epoch, history)
    plot_metrics(epochs, history)
    plt.savefig('curve_fig/animal_blood/Transmission_animalBlood_preprocessing.jpg')
    plt.show()


    # test
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    
    test_precision = tf.keras.metrics.Precision(name='precision')
    test_recall = tf.keras.metrics.Recall(name='recall')
    test_auc = tf.keras.metrics.AUC(name='auc')
    test_prc = tf.keras.metrics.AUC(name='prc', curve='PR')

    test_acc_1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy')
    test_acc_3 = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    test_acc_5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')

    @tf.function
    def test_step(test_spectral, test_labels):
        output = model(test_spectral, training=False)
        loss = loss_object(test_labels, output)

        test_loss(loss)
        test_accuracy(test_labels, output)

        test_precision(test_labels, output)
        test_recall(test_labels, output)
        test_auc(test_labels, output)
        test_prc(test_labels, output)

        test_acc_1(test_labels, output)
        test_acc_3(test_labels, output)
        test_acc_5(test_labels, output)

        return output

    test_bar = tqdm(test_ds, file=sys.stdout)
    y_true = np.array([])
    y_pred = np.array([])
    for spectrals, labels in test_bar:
        output = test_step(spectrals, labels)
        y_true = np.append(y_true, labels.numpy())
        y_pred = np.append(y_pred, output.numpy())
        # print val process
        test_bar.desc = "test: loss:{:.4f}, acc:{:.4f}, precision:{:.4f}, recall:{:.4f}, auc:{:.4f}, prc:{:.4f}, acc_1:{:.4f}, acc_3:{:.4f}, acc_5:{:.4f}".format(
                                                                            test_loss.result(),
                                                                            test_accuracy.result(),
                                                                            test_precision.result(),
                                                                            test_recall.result(),
                                                                            test_auc.result(),
                                                                            test_prc.result(),
                                                                            test_acc_1.result(),
                                                                            test_acc_3.result(),
                                                                            test_acc_5.result(),
                                                                            )

    F1_score = 2 * (test_precision.result() * test_recall.result()) / (test_precision.result() + test_recall.result())
    print('F1_score: {:.4f}'.format(F1_score))


    y_true = y_true.reshape(-1, num_classes).tolist()
    y_pred = y_pred.reshape(-1, num_classes).tolist()

    plot_roc(y_true, y_pred, num_classes)
    plot_prc(y_true, y_pred, num_classes)

    for i in range(len(y_pred)):
        max_value=max(y_pred[i])
        if max_value == 0:
            print(i, ":")
            print("y_true:")
            print(y_true[i])
            print("y_pred:")
            print(y_pred[i])

        for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0

    plot_cm(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    np.set_printoptions(threshold=np.inf)
    # print(y_true)
    y_true_categorical = np.zeros([42,42])
    for i in range(0, 42):
        y_true_categorical[i][i] = 1
    y_true_1 = lb.inverse_transform(y_true_categorical)
    print(y_true_1)

    print('Classification_report', sklearn_metrics.classification_report(y_true, y_pred, digits=4, zero_division=1))
    print('accuracy_score', sklearn_metrics.accuracy_score(y_true, y_pred))
    print('------Weighted------')
    print('Weighted precision', precision_score(y_true, y_pred, average='weighted', zero_division=1))
    print('Weighted recall', recall_score(y_true, y_pred, average='weighted', zero_division=1))
    print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted', zero_division=1))
    


if __name__ == "__main__":
    main()
