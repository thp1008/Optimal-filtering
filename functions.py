import math
import numpy as np

def toysim(x):
    N1=10.0
    mu=4
    sigma=5
    tau=.5
    t_naught=(sigma**2)/(2*tau)
    N2=N1*math.exp(-((t_naught-mu)/sigma)**2+(t_naught/tau))
    if x<=t_naught:
        y=N1*math.exp(-((x-mu)/sigma)**2)
    else:
        y=N2*math.exp(-x/tau)
    return y

def generate_data(n_samples):
    x_data=np.linspace(0,15,n_samples)
    y_data=np.zeros(n_samples)
    for i in range(len(x_data)):
        y_data[i]=toysim(x_data[i])
    return x_data,y_data

def noisemaker(dataset,intensity):
    setsize=np.size(dataset)
    randset=np.random.normal(scale=intensity,size=setsize)
    return dataset+randset

def amp_matrix(n_samples,pulse_model):
    M=np.ones((n_samples,2))
    M[:,0]=pulse_model
    MT=M.transpose()
    R=np.diag([1]*n_samples)
    invR=np.linalg.inv(R)
    block1=np.linalg.inv(np.dot(MT,np.dot(invR,M)))
    block2=np.dot(block1,np.dot(MT,invR))
    return block2

def find_amp(dataset,block2):
    p_hat=np.dot(block2,dataset)
    amplitude=p_hat[0]
    return amplitude

def find_blip(intensity,n_samples,pulse_model):
    blip=np.zeros(len(pulse_model))
    block2=amp_matrix(n_samples,pulse_model)
    for i in range(len(blip)):
        x_data,y_data=generate_data(n_samples)
        measured_data=noisemaker(y_data,intensity)
        blip[i]=(1-find_amp(measured_data,block2))
    return blip