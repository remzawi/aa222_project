#File containing functions required to perform optimization
import numpy as np 
import numpy.random as rd 
from kerasmodel import *


#Contains parameters to optimize, associated with a bound (if the bounds are the same value, no optimization is done) and a flag for value types (0 normal, 1 int, 2 log)
params={'learning_rate':(1e-6,1e-1,2),
        'batch_size':(16,256,1),
        'nepochs':(5,5,1),
        'conv_size1':(64,64,1),
        'conv_size2':(64,64,1),
        'fc_size':(50,200,1),
        'dropout_param':(0,1,0),
        'l2_reg':(1e-10,1,2)}

def getParamsToOptimize(params=params):
    paramstooptimize=[]
    bounds=[]
    model_params={}
    for param,bound in params.items():
        if bound[0]!=bound[1]:
            paramstooptimize.append(param)
            bounds.append(bound)
        else:
            model_params[param]=bound[0]
    return paramstooptimize,bounds ,model_params




def createRandomSamplingPlan(n,bounds): #Create n points in len(bounds) dim that for each dim fall between bounds[i] bounds. If ints[i] is true then sample integers    
    ndims=len(bounds)
    samples=np.zeros((ndims,n))
    for i in range(ndims):
        if bounds[i][2]==1:
            samples[i]=rd.randint(bounds[i][0],bounds[i][1]+1,n)
        elif bounds[i][2]==0:
            samples[i]=rd.uniform(bounds[i][0],bounds[i][1],n)
        else:
            samples[i]=np.exp(rd.uniform(np.log(bounds[i][0]),np.log(bounds[i][1]),n))
    return samples.T

def randomSearchFromSamples(samples,paramstooptimize,model_params,num_training=49000,num_val=1000):
    best_model=None
    best_score=0
    best_params=None
    X_train,y_train,X_val,y_val,X_test,y_test=load_cifar(num_training=num_training,num_val=num_val)
    n,m=samples.shape
    for i in range(n):
        print("Starting training on sample "+str(i+1))
        for j in range(m):
            model_params[paramstooptimize[j]]=samples[i,j]
        model=create_model(model_params)
        score=score_model(model,model_params,X_train,y_train,X_val,y_val)
        if score>best_score:
            best_model=model
            best_params=model_params.copy()
            best_score=score
    return best_model,best_params,best_score

def randomSearch(n,test_sampling=False,params=params,num_training=49000,num_val=1000):
    paramstooptimize,bounds ,model_params=getParamsToOptimize(params)
    samples=createRandomSamplingPlan(n,bounds)
    if test_sampling:
        return samples
    return randomSearchFromSamples(samples,paramstooptimize,model_params,num_training,num_val)

model =randomSearch(1)



