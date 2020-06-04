#File containing functions required to perform optimization
import numpy as np 
import numpy.random as rd 
from kerasmodel import *
from tqdm import tqdm
import gc


#Contains parameters to optimize, associated with a bound (if the bounds are the same value, no optimization is done) and a flag for value types (0 normal, 1 int, 2 log)
params={'learning_rate':(1e-5,1e-1,2),
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
    best_params,y_best=None,0
    X_train,y_train,X_val,y_val,X_test,y_test=load_cifar(num_training=num_training,num_val=num_val)
    n,m=samples.shape
    params_history,y_history=[],[]
    if keras_verbose>0:
        for i in range(n):
            print("Starting training on sample "+str(i+1))
            for j in range(m):
                model_params[paramstooptimize[j]]=samples[i,j]
            y=score_modelv3(model_params,X_train,y_train,X_val,y_val,keras_verbose=keras_verbose)
            if y>y_best:
                best_params=model_params.copy()
                params_history.append(model_params.copy())
                y_best=y
                y_history.append(y)
    else:
        for i in tqdm(range(n)):
            for j in range(m):
                model_params[paramstooptimize[j]]=samples[i,j]
            y=score_modelv3(model_params,X_train,y_train,X_val,y_val,keras_verbose=keras_verbose)
            if y>y_best:
                best_params=model_params.copy()
                params_history.append(model_params.copy())
                y_best=y
                y_history.append(y)
    return best_params,y_best,params_history,y_history

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
    def update(self,x_best,w_factor=1):
        r1,r2=rd.rand(self.n),rd.rand(self.n)
        self.w*=w_factor
        self.x+=self.v
        self.x=correct(self.x,self.bounds)    
        self.v=self.w*self.v+self.c1*r1*(self.x_best-self.x)+self.c2*r2*(x_best-self.x)

def correct(x,bounds):
    for i in range(len(x)):
        x[i]=min(x[i],bounds[i][1])
        x[i]=max(x[i],bounds[i][0])
        if bounds[i][2]==1:
            x[i]=int(x[i])
    return x
    
def createPopulation(n,bounds,w=1,c1=1,c2=1):
    samples=createRandomSamplingPlan(n,bounds)
    population=[]
    for i in range(n):
        particle=Particle(samples[i],bounds,w,c1,c2)
        population.append(particle)
    return population

def PS_optimization(f,population,k_max,w=1.2,c1=1.5,c2=1.5,progress=True,w_update=True):
    x_best,y_best=population[0].x_best,0
    history_x=[]
    history_y=[]
    w_factor=1
    if progress:
        print("Initial evaluation on population")
        for p in population:
            y=f(p.x)
            p.y_best=y
            if y>y_best:
                x_best,y_best=p.x.copy(),y
                history_x.append(x_best.copy())
                history_y.append(y_best)
        print("Starting optimization loop")
        for k in tqdm(range(k_max)):
            if w_update and (k==k_max//4 or k==k_max//2 or k_max==3*k_max//4):
                w_factor=0.8
            else:
                w_factor=1
            for p in population:
                p.update(x_best,w_factor)
                y=f(p.x)
                if y>p.y_best:
                    p.x_best=p.x.copy() 
                    p.y_best=y
                if y>y_best:
                    x_best=p.x.copy()
                    y_best=y
                    history_x.append(x_best.copy())
                    history_y.append(y_best)
        return population,x_best,y_best,history_x,history_y
    else:
        for p in population:
            y=f(p.x)
            p.y_best=y
            if y>y_best:
                x_best,y_best=p.x.copy(),y
                history_x.append(x_best.copy())
                history_y.append(y_best)

        for k in range(k_max):
            if w_update and (k==k_max//4 or k==k_max//2 or k_max==3*k_max//4):
                w_factor=0.8
            else:
                w_factor=1
            for p in population:
                p.update(x_best,w_factor)
                y=f(p.x)
                if y>p.y_best:
                    p.x_best=p.x.copy() 
                    p.y_best=y
                if y>y_best:
                    x_best=p.x.copy() 
                    y_best=y
                    history_x.append(x_best.copy())
                    history_y.append(y_best)
        return population,x_best,y_best,history_x,history_y

def particle_swarm(n,k_max,params=params,w=1.2,c1=2,c2=2,num_training=49000,num_val=1000,keras_verbose=2,w_update=True):
    paramstooptimize,bounds ,model_params=getParamsToOptimize(params)
    population=createPopulation(n,bounds)
    X_train,y_train,X_val,y_val,X_test,y_test=load_cifar(num_training=num_training,num_val=num_val)
    def f(x):
        mp=model_params.copy()
        for j in range(len(x)):
            mp[paramstooptimize[j]]=x[j]
        return score_modelv3(mp,X_train,y_train,X_val,y_val,keras_verbose=keras_verbose)
    population,x_best,y_best,history_x,history_y=PS_optimization(f,population,k_max,w=w,c1=c1,c2=c2,progress=keras_verbose==0,w_update=w_update)
    for j in range(len(x_best)):
        model_params[paramstooptimize[j]]=x_best[j]
    return model_params,y_best,history_x,history_y
    
def crossover(x1,x2,a=0.15): #BLX alpha crossover
    low=np.minimum(x1,x2)
    up=np.maximum(x1,x2)
    low,up=low-a*(up-low),up+a*(up-low)
    x=low+rd.rand(len(x1))*(up-low)
    return x


def mutation(x,scale=0.1,relative=True): #if normalize apply scale relative to value range
    if relative:
        return x*(1+rd.normal(scale=scale,size=len(x)))
    return x+rd.normal(scale=scale,size=len(x))


def tournament_selection(population,y,k):
    parents=[]
    for i in range(len(y)):
        mask=rd.randint(0,len(y),k)
        sub_pop=population[mask]
        y_sp=y[mask]
        order=np.argsort(y_sp)
        parents.append((sub_pop[order[-1]],sub_pop[order[-2]]))
    return parents

def correct_population(population,bounds):
    for i in range(len(population)):
        population[i]=correct(population[i],bounds)
    return population

def genetic_algorithm(f,population,k_max,k_selection,bounds,selection=tournament_selection,crossover=crossover,mutation=mutation,progress=True):
    if progress:
        n=len(population)
        y=np.zeros(n)
        x_history,y_history=[],[]
        y_best=0
        x_best=None
        print("Starting optimization loop")
        for k in tqdm(range(k_max)):
            for i in range(n):
                y[i]=f(population[i])
            ind=np.argmax(y)
            if y[ind]>y_best:
                y_best=y[ind]
                x_best=population[ind]
                x_history.append(population[ind])
                y_history.append(y[ind])
            parents=selection(population,y,k_selection)
            children=[crossover(parents[i][0],parents[i][1]) for i in range(n)]
            population=np.array([mutation(children[i]) for i in range(n)])
            population=correct_population(population,bounds)
        print("Final evaluation")
        for i in range(n):
            y[i]=f(population[i])
        ind=np.argmax(y)
        if y[ind] > y_best:
            y_best=y[ind]
            x_best=population[ind]
            x_history.append(population[ind])
            y_history.append(y[ind])
        return x_best,y_best,x_history,y_history
    else:
        n=len(population)
        y=np.zeros(n)
        x_history,y_history=[],[]
        y_best=0
        x_best=None
        for k in range(k_max):
            for i in range(n):
                y[i]=f(population[i])
            ind=np.argmax(y)
            if y[ind]>y_best:
                x_best=population[ind]
                y_best=y[ind]
                x_history.append(population[ind])
                y_history.append(y[ind])
            parents=selection(population,y,k_selection)
            children=[crossover(parents[i][0],parents[i][1]) for i in range(n)]
            population=np.array([mutation(children[i]) for i in range(n)])
            population=correct_population(population,bounds)
        for i in range(n):
            y[i]=f(population[i])
        ind=np.argmax(y)
        if y[ind] > y_best:
            y_best=y[ind]
            x_best=population[ind]
            x_history.append(population[ind])
            y_history.append(y[ind])
        return x_best,y_best,x_history,y_history
    


def GP(n,k_max,k_selection,params=params,num_training=49000,num_val=1000,keras_verbose=2):
    paramstooptimize,bounds ,model_params=getParamsToOptimize(params)
    population=createRandomSamplingPlan(n,bounds)
    X_train,y_train,X_val,y_val,X_test,y_test=load_cifar(num_training=num_training,num_val=num_val)
    def f(x):
        mp=model_params.copy()
        for j in range(len(x)):
            mp[paramstooptimize[j]]=x[j]
        return score_modelv3(mp,X_train,y_train,X_val,y_val,keras_verbose=keras_verbose)
    x_best,y_best,x_history,y_history=genetic_algorithm(f,population,k_max,k_selection,bounds,progress=keras_verbose==0)
    for j in range(len(x_best)):
        model_params[paramstooptimize[j]]=x_best[j]
    return model_params,y_best,x_history,y_history
        






