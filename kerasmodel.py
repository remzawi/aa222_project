#Contains the base keras model, a function that builds a model using the given hyperparamters, and a function to evaluate 1 model
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, ReLU, Flatten
import numpy as np

def load_cifar(num_training=49000,num_val=1000,normalize=True): #Load data from saved numpy array
    X_train=np.asarray(np.load('X_train_cifar10.npy'),dtype=np.float32)
    X_test=np.asarray(np.load('X_test_cifar10.npy'),dtype=np.float32)
    y_train=np.asarray(np.load('y_train_cifar10.npy'),dtype=np.int32).flatten()
    y_test=np.asarray(np.load('y_test_cifar10.npy'),dtype=np.int32).flatten()
    X_train,X_val=X_train[:num_training],X_train[num_training:num_training+num_val]
    y_train,y_val=y_train[:num_training],y_train[num_training:num_training+num_val]
    if normalize:
        mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        std = X_train.std(axis=(0, 1, 2), keepdims=True)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
    return X_train, y_train, X_val, y_val, X_test, y_test 



def score_model(model,model_params,X_train,y_train,X_val,y_val): #Black box function that scores a model by returning its final validation accuracy
    lr=model_params['learning_rate']
    nepochs=model_params['nepochs']
    batch_size=['batch_size']
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    history=model.train(X_train,y_train,batch_size=batch_size,epochs=nepochs,validation_data=(X_val,y_val),verbose=False)
    score=history.history['val_sparse_categorical_accuracy'][-1]
    return score

#Assume CONV-Batchnorm-CONV-Batchnorm-CONV-Batchnorm-POOL-Dropout-FC-Softmax
def create_model(model_params):
    conv_size1,conv_size2,conv_size3=model_params['conv_size1'],model_params['conv_size2'],model_params['conv_size3'] #Only depth, assumes stride one, size 5 and same padding
    fc_size=model_params['fc_size']
    dropout_param=model_params['dropout_param']
    reg=tf.keras.regularizers.l2(l=model_params['l2_reg'])
    model=tf.keras.Sequential()
    model.add(Conv2D(conv_size1,3,padding='same',kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(conv_size2,3,padding='same',kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(conv_size3,3,padding='same',kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(dropout_param))
    model.add(Dense(fc_size,kernel_regularizer=reg))
    model.add(ReLU())
    model.add(Dense(10,activation='softmax',kernel_regularizer=reg))
    return model








