#File containing functions required to perform optimization
import numpy as np 
import numpy.random as rd 
from kerasmodel import *
from tqdm import tqdm


#Contains parameters to optimize, associated with a bound (if the bounds are the same value, no optimization is done) and a flag for value types (0 normal, 1 int, 2 log)
params={'learning_rate':(1e-4,1e-1,2),
        'batch_size':(32,256,1),
        'nepochs':(8,8,1),
        'conv_size1':(32,128,1),
        'conv_size2':(32,128,1),
        'conv_size3':(32,128,1),
        'fc_size':(50,200,1),
        'dropout_param':(0,0.8,0),
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

def randomSearchFromSamples(samples,paramstooptimize,model_params,num_training=49000,num_val=1000,keras_verbose=2):
    best_model=None
    best_score=0
    best_params=None
    X_train,y_train,X_val,y_val,X_test,y_test=load_cifar(num_training=num_training,num_val=num_val)
    n,m=samples.shape
    if keras_verbose>0:
        for i in range(n):
            print("Starting training on sample "+str(i+1))
            for j in range(m):
                model_params[paramstooptimize[j]]=samples[i,j]
            model=create_model(model_params)
            score=score_model(model,model_params,X_train,y_train,X_val,y_val,keras_verbose=keras_verbose)
            if score>best_score:
                best_model=model
                best_params=model_params.copy()
                best_score=score
    else:
        for i in tqdm(range(n)):
            for j in range(m):
                model_params[paramstooptimize[j]]=samples[i,j]
            model=create_model(model_params)
            score=score_model(model,model_params,X_train,y_train,X_val,y_val,keras_verbose=keras_verbose)
            if score>best_score:
                best_model=model
                best_params=model_params.copy()
                best_score=score
    return best_model,best_params,best_score

def randomSearch(n,test_sampling=False,params=params,num_training=49000,num_val=1000,keras_verbose=2):
    paramstooptimize,bounds ,model_params=getParamsToOptimize(params)
    samples=createRandomSamplingPlan(n,bounds)
    if test_sampling:
        return samples
    return randomSearchFromSamples(samples,paramstooptimize,model_params,num_training,num_val,keras_verbose=keras_verbose)

class Particle():
    def __init__(self,x,bounds,w,c1,c2):
        self.x=x
        self.n=len(x)
        self.x_best=x
        self.y_best=np.inf
        self.v=np.zeros(len(x))
        self.bounds=bounds
        self.w=w
        self.c1=c1 
        self.c2=c2
    def update(self,x_best):
        r1,r2=rd.rand(self.n),rd.rand(self.n)
        self.x+=self.v
        for i in range(self.n):
            self.x[i]=min(self.x[i],self.bounds[i][1])
            self.x[i]=max(self.x[i],self.bounds[i][0])
            if self.bounds[i][2]==1:
                self.x[i]=int(self.x[i])     
        self.v=self.w*self.v+self.c1*r1*(self.x_best-self.x)+self.c2*r2*(x_best-self.x)
    
def createPopulation(n,bounds,w=1,c1=1,c2=1):
    samples=createRandomSamplingPlan(n,bounds)
    population=[]
    for i in range(n):
        particle=Particle(samples[i],bounds,w,c1,c2)
        population.append(particle)
    return population

def PS_optimization(f,population,k_max,w=1,c1=1,c2=1,progress=True):
    x_best,y_best=population[0].x_best.copy(),0
    if progress:
        print("Initila evaluation on population")
        for p in population:
            y=f(p.x)
            p.y_best=y
            if y>y_best:
                x_best,y_best=p.x.copy(),y
        print("Starting optimization loop")
        for k in tqdm(range(k_max)):
            for p in population:
                p.update(x_best)
                y=f(p.x)
                if y>p.y_best:
                    p.x_best=p.x 
                    p.y_best=y
                if y>y_best:
                    x_best=p.x 
                    y_best=y
        return population,x_best,y_best
    else:
        for p in population:
            y=f(p.x)
            p.y_best=y
            if y>y_best:
                x_best,y_best=p.x.copy(),y

        for k in range(k_max):
            for p in population:
                p.update(x_best)
                y=f(p.x)
                if y>p.y_best:
                    p.x_best=p.x 
                    p.y_best=y
                if y>y_best:
                    x_best=p.x 
                    y_best=y
        return population,x_best,y_best

def particle_swarm(n,k_max,params=params,w=1,c1=1,c2=1,num_training=49000,num_val=1000,keras_verbose=2):
    paramstooptimize,bounds ,model_params=getParamsToOptimize(params)
    population=createPopulation(n,bounds)
    X_train,y_train,X_val,y_val,X_test,y_test=load_cifar(num_training=num_training,num_val=num_val)
    def f(x):
        for j in range(len(x)):
            model_params[paramstooptimize[j]]=x[j]
        model=create_model(model_params)
        score=score_model(model,model_params,X_train,y_train,X_val,y_val,keras_verbose=keras_verbose)
        return score
    population,x_best,y_best=PS_optimization(f,population,k_max,w=w,c1=c1,c2=c2,progress=keras_verbose==0)
    for j in range(len(x_best)):
        model_params[paramstooptimize[j]]=x_best[j]
    return model_params,y_best
    



