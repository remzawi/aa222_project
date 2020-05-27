#Contains the base keras model, a function that builds a model using the given hyperparamters, and a function to evaluate 1 model
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, ReLU, Flatten
from tensorflow.keras.backend import clear_session
import numpy as np
import gc


class ToyNet(tf.keras.Model):
    def __init__(self,model_params):
        super(ToyNet, self).__init__()
        conv_size1,conv_size2,conv_size3=int(model_params['conv_size1']),int(model_params['conv_size2']),int(model_params['conv_size3']) #Only depth, assumes stride one, size 5 and same padding
        fc_size=int(model_params['fc_size'])
        dropout_param=model_params['dropout_param']
        reg=tf.keras.regularizers.l2(model_params['l2_reg'])
        self.relu=ReLU()
        self.pool=MaxPooling2D()
        self.flatten=Flatten()
        self.dropout=Dropout(dropout_param)
        self.c2d1=Conv2D(conv_size1,3,padding='same',kernel_regularizer=reg)
        self.c2d2=Conv2D(conv_size2,3,padding='same',kernel_regularizer=reg)
        self.c2d3=Conv2D(conv_size3,3,padding='same',kernel_regularizer=reg)
        self.fc=Dense(fc_size,kernel_regularizer=reg)
        self.softmax=Dense(10,activation='softmax',kernel_regularizer=reg)
        self.bn1=BatchNormalization()
        self.bn2=BatchNormalization()
        self.bn3=BatchNormalization()

    
    def call(self, input_tensor, training=False):
        x=self.c2d1(input_tensor)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.c2d2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.c2d3(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.flatten(x)
        x=self.dropout(x)
        x=self.fc(x)
        x=self.relu(x)
        x=self.softmax(x)

        return x

def load_cifar(num_training=49000,num_val=1000,normalize=True): #Load data from saved numpy array
    X_train=np.asarray(np.load('X_train_cifar10.npy'),dtype=np.float32)/255
    X_test=np.asarray(np.load('X_test_cifar10.npy'),dtype=np.float32)/255
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



def score_model(model,model_params,X_train,y_train,X_val,y_val,keras_verbose=2): #Black box function that scores a model by returning its final validation accuracy
    lr=model_params['learning_rate']
    nepochs=int(model_params['nepochs'])
    batch_size=int(model_params['batch_size'])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    np.random.seed(0)
    history=model.fit(X_train,y_train,batch_size=batch_size,epochs=nepochs,validation_data=(X_val,y_val),verbose=keras_verbose)
    score=history.history['val_sparse_categorical_accuracy'][-1]
    del model
    gc.collect()
    return score

#Assume CONV-Batchnorm-Relu-CONV-Batchnorm-Relu-CONV-Batchnorm-Relu-POOL-Dropout-FC-Softmax
def create_model(model_params):
    conv_size1,conv_size2,conv_size3=int(model_params['conv_size1']),int(model_params['conv_size2']),int(model_params['conv_size3']) #Only depth, assumes stride one, size 5 and same padding
    fc_size=int(model_params['fc_size'])
    dropout_param=model_params['dropout_param']
    reg=tf.keras.regularizers.l2(model_params['l2_reg'])
    model=tf.keras.Sequential()
    model.add(Conv2D(conv_size1,3,padding='same',kernel_regularizer=reg,input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(conv_size2,3,padding='same',kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D())
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

def score_modelv2(model_params,X_train,y_train,X_val,y_val,keras_verbose=2):
    return score_model(create_model(model_params),model_params,X_train,y_train,X_val,y_val,keras_verbose)

def score_modelv3(model_params,X_train,y_train,X_val,y_val,keras_verbose=2):
    clear_session()
    model=ToyNet(model_params)
    lr=model_params['learning_rate']
    nepochs=int(model_params['nepochs'])
    batch_size=int(model_params['batch_size'])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    history=model.fit(X_train,y_train,batch_size=batch_size,epochs=nepochs,validation_data=(X_val,y_val),verbose=keras_verbose)
    score=history.history['val_sparse_categorical_accuracy'][-1]
    clear_session()
    return score
    

def evaluate_model(model):
    try:
        X_test=np.load('X_test_cifar10.npy')
        y_test=np.load('y_test_cifar10.npy')
        return model.evaluate(X_test,y_test)
    except:
        print("CIFAR data not downloaded!")
    








