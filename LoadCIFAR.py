#This script downloards cifar10 dataset and saves it as a numpy file

from tensorflow.keras.datasets import cifar10
import numpy as np 
import os

def loadCIFAR():
    if os.path.exists('X_train_cifar10.npy'):
        print("Already loaded.")
    else:
        (X_train, y_train), (X_test, y_test)=cifar10.load_data()
        np.save('X_train_cifar10.npy',X_train)
        np.save('X_test_cifar10.npy',X_test)
        np.save('y_train_cifar10.npy',y_train)
        np.save('y_test_cifar10.npy',y_test)
        print("Successfully loaded CIFAR10 dataset.")
    return




