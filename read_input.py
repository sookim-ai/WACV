import tensorflow as tf
from function import *
import numpy as np
import random




def read_input(path):
    image=np.load("./X_light.npy")
    image=image[:,:,:,:,:,0:3]
    label=np.load("./Y_light.npy")
    d,__,__,__,__,__ = np.shape(image)
    te_image=np.asarray(image[0:int(d*0.2)])
    te_label=np.asarray(label[0:int(d*0.2)])
    tr_image=np.asarray(image[int(d*0.20):d-10])
    tr_label=np.asarray(label[int(d*0.20):d-10])
    va_image=np.asarray(image[d-10:d])
    va_label=np.asarray(label[d-10:d])
    np.save("X_test.npy",te_image)
    np.save("Y_test.npy",te_label)
    print(np.shape(tr_image),np.shape(tr_label),np.shape(te_image),np.shape(te_label),np.shape(va_image),np.shape(va_label))
    return tr_image,tr_label,te_image,te_label,va_image,va_label
