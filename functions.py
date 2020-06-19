import math
import numpy as np

def toysim(x,mu=3):
    #mu is the peak amplitude
    N1=10.0
    sigma=.3
    tau=1.5
    t_naught=((sigma**2)/(2*tau))+mu
    N2=N1*math.exp(-((t_naught-mu)/sigma)**2+(t_naught/tau))
    if x<=t_naught:
        y=N1*math.exp(-((x-mu)/sigma)**2)
    else:
        y=N2*math.exp(-x/tau)
    return y

def generate_data(n_samples,peak=4):
    x_data=np.linspace(0,10,n_samples)
    y_data=np.zeros(n_samples)
    for i in range(len(x_data)):
        y_data[i]=toysim(x_data[i],mu=peak)
    return x_data,y_data

def noisemaker(dataset,intensity=1):
    setsize=np.size(dataset)
    randset=np.random.normal(scale=intensity,size=setsize)
    newset=dataset+randset
    return newset

def generate_pulsemod(pulse):
    pulsemod=np.zeros(len(pulse))
    for i in range(1000):
        pulsemod+=noisemaker(pulse)
    return pulsemod/1000

def amp_matrix(pulse_model):
    M=np.ones((len(pulse_model),2))
    M[:,0]=pulse_model
    MT=M.transpose()
    R=np.diag([1]*len(pulse_model))
    invR=np.linalg.inv(R)
    block1=np.linalg.inv(np.dot(MT,np.dot(invR,M)))
    block2=np.dot(block1,np.dot(MT,invR))
    return block2

def find_amp(dataset,block2):
    p_hat=np.dot(block2,dataset)
    amplitude=p_hat[0]
    return amplitude

def find_blip(intensity,pulse_model,trials=300):
    blip=np.zeros(trials)
    block2=amp_matrix(pulse_model)
    for i in range(trials):
        x_data,y_data=generate_data(len(pulse_model))
        measured_data=noisemaker(y_data,intensity)
        blip[i]=(1-find_amp(measured_data,block2))
    return blip