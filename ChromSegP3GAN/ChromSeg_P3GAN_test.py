from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import load_model

# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# load image data
dataset_org= load_real_samples('crossing_touching_G banded chromsome_data.npz')
print('Loaded', dataset_org[0].shape, dataset_org[1].shape)

temp0= dataset_org[0]
temp1 = dataset_org[1]
print('Loaded', temp0.shape,temp1.shape)

d1 = temp0[0:400,:,:,:]
d2 = temp1[0:400,:,:,:]
dataset = (d1,d2)

print('Loaded', d1.shape, d2.shape)

print('Loaded', dataset[0].shape, dataset[1].shape)

t1 = temp0[400: 500,:,:,:]
t2 = temp1[400: 500,:,:,:]
test = (t1,t2)
print('Loaded', t1.shape,t2.shape)

# load model
g_model = load_model('./model/model_040000.h5')

# Few useful metrics and losses
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / ((K.sum(y_true_f) + K.sum(y_pred_f) - intersection) + 1.0)


def precision(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec

def recall(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall


def true_positive(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
    return tp

def true_negative(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    return tn


def false_positive(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = (K.sum(y_neg * y_pred_pos) + smooth) / (K.sum(y_neg) + smooth)
    return tp

def false_negative(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_pos * y_pred_neg) + smooth) / (K.sum(y_pos) + smooth)
    return tn


def accuracy(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    tn = K.sum(y_neg * y_pred_neg)
    acc = (tp+tn + smooth) / (tp + tn + fn + fp + smooth)
    return acc


dice = []
iou = []

rec= []
prec = []

tpr_data = []
fpr_data = []
fnr_data = []
tnr_data = []
acc_data = []

predictions = g_model.predict(t1[0:100,:,:,:])

for i in range(len(predictions)):
    predictions[i] = (predictions[i]+1)/2
    t2[i:i+1,:,:,:] = (t2[i:i+1,:,:,:][0]+1)/2
    
   
    iou.append(jacard_coef(predictions[i],t2[i:i+1,:,:,:]))
    dice.append(dice_coef(predictions[i],t2[i:i+1,:,:,:]))
   
    rec.append(recall(predictions[i],t2[i:i+1,:,:,:]))
    prec.append(precision(predictions[i],t2[i:i+1,:,:,:])) 
    tpr_data.append(true_positive(predictions[i],t2[i:i+1,:,:,:]))
    fpr_data.append(false_positive(predictions[i],t2[i:i+1,:,:,:]))
    tnr_data.append(true_negative(predictions[i],t2[i:i+1,:,:,:]))
    fnr_data.append(false_negative(predictions[i],t2[i:i+1,:,:,:]))
    acc_data.append(accuracy(predictions[i],t2[i:i+1,:,:,:]))


print("Accuracy : ", np.mean(acc_data))
print("Precision : ", np.mean(prec))
print("Recall : ", np.mean(rec))
print("TPR : ", np.mean(tpr_data))
print("FPR: ", np.mean(fpr_data))
print("FNR: ", np.mean(fnr_data))
print("TNR: ", np.mean(tnr_data))
print("IOU: ", np.mean(iou))
print("DICE: ", np.mean(dice))