import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def model_corr(file,spike,spikewidth=50):
    #makes the pulse model and finds the autocorrelation matrix
    traces=np.zeros(1285)
    sqr=np.zeros(1285)
    count=0
    amplitudes1=np.zeros(0)
    for i in range(file.nPulses):
        tr=file.read_trace(i)
        ave=np.average(tr[0:500])
        amp=np.amax(tr)-ave
        tr=tr[515:1800]
        if abs(amp-spike) <= spikewidth:
            traces=traces+(tr-ave)
            count+=1
            sqr=sqr+(tr-ave)**2
            amplitudes1=np.append(amplitudes1,amp)
    pulsemodel=traces/count
    corr_matrix=sqr/count-(pulsemodel)**2
    return pulsemodel,corr_matrix,amplitudes1

def amp_matrix(pulse_model,corr_matrix):
    M=np.ones((len(pulse_model),2))
    M[:,0]=pulse_model
    MT=M.transpose()
    R=np.diag(corr_matrix)
    invR=np.linalg.inv(R)
    block1=np.linalg.inv(np.dot(MT,np.dot(invR,M)))
    block2=np.dot(block1,np.dot(MT,invR))
    return block2

def find_spike(file):
    amplitudes=np.zeros(file.nPulses)
    for i in range(file.nPulses):
        tr=file.read_trace(i)
        ave=np.average(tr[0:500])
        tr=tr[515:1800]
        amp=np.amax(tr)-ave
        amplitudes[i]=amp
    counts,bins=np.histogram(amplitudes,bins=200,range=(0,10000))
    spike=(bins[np.argmax(counts)-1]+bins[np.argmax(counts)+1])/2
    # spikewidth=abs(spike-bins[np.argmax(counts)])
    spikewidth=100
    return spike,spikewidth

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

def calibration(pars1,pars2,pars3,pars4):
    slope1=-.6403/(pars1[0]-pars2[0])
    slope2=-.6403/(pars3[0]-pars4[0])
    offset1=41.5427-pars2[0]*slope1        
    offset2=41.5427-pars4[0]*slope2    
    energyres1=pars1[1]*slope1
    energyres2=pars3[1]*slope2
    energyres3=pars2[1]*slope1
    energyres4=pars4[1]*slope2
    improvement=energyres2/energyres1
    return improvement

def gauss(x,mu,sig,scl):
    return scl*norm.pdf(x, loc=mu, scale=sig)

def improv_graph(f):
    spike,spikewidth=find_spike(f)
    pulsemodel,corr_matrix,amplitudes1=model_corr(f,spike)
    block2=amp_matrix(pulsemodel,corr_matrix)
    amplitudes2=np.zeros(0)
    for i in range(f.nPulses):
        tr=f.read_trace(i)
        ave=np.average(tr[0:500])
        amp=np.amax(tr)-ave
        tr=tr-ave
        tr=tr[515:1800]
        scale=find_amp(tr,block2)
        amp1=scale*np.amax(pulsemodel)
        if abs(amp-spike) <= spikewidth:
            amplitudes2=np.append(amplitudes2,amp1)
    countsa,binsa=np.histogram(amplitudes1,bins=90)
    countsb,binsb=np.histogram(amplitudes2,bins=binsa)
    centers=(0.5*(binsa[1:]+binsa[:-1]))
    
    # range2=np.arange(len(centers)//2,len(centers))
    
    guess1apos=np.argmax(countsa)
    pars1,cov1=curve_fit(gauss, centers, countsa, p0=[centers[guess1apos],6,guess1apos])
    pars1,cov1=curve_fit(gauss, centers, countsa, p0=[centers[guess1apos],pars1[1],pars1[0]])
    cut1=pars1[0]-2*pars1[1]
    for i,j in enumerate(centers):
        if abs(j-cut1)<=2:
            cut1=i
            break
    guess2apos=np.argmax(countsa[0:cut1])
    pars2,cov2=curve_fit(gauss, centers[0:cut1], countsa[0:cut1], p0=[centers[0:cut1][guess2apos],6,guess2apos])
    pars2,cov2=curve_fit(gauss, centers[0:cut1], countsa[0:cut1], p0=[centers[0:cut1][guess2apos],pars2[1],pars2[0]])
    
    guess1bpos=np.argmax(countsb)
    pars3,cov3=curve_fit(gauss, centers, countsb, p0=[centers[guess1bpos],6,guess1bpos])
    pars3,cov3=curve_fit(gauss, centers, countsb, p0=[centers[guess1bpos],pars3[1],pars3[0]])
    cut2=pars3[0]-2*pars3[1]
    for i,j in enumerate(centers):
        if abs(j-cut2)<=2:
            cut2=i
            break
    guess2bpos=np.argmax(countsb[0:cut2])
    pars4,cov4=curve_fit(gauss, centers[0:cut2], countsb[0:cut2], p0=[centers[0:cut2][guess2bpos],6,guess2bpos])
    pars4,cov4=curve_fit(gauss, centers[0:cut2], countsb[0:cut2], p0=[centers[0:cut2][guess2bpos],pars4[1],pars4[0]])
    
    
    # guess1bpos=np.argmax(countsb)
    # a2=np.argmax(countsa[range2])

    
    # pars3,cov3=curve_fit(gauss, centers[range1], countsb[range1], p0=[centers[range1][a3],4,a3])
    # pars4,cov4=curve_fit(gauss, centers[range2], countsb[range2], p0=[centers[range2][a4],4,a4])
    # pars3=(centers[range1][a3],4,a3)    
    # pars4=(centers[range2][a4],4,a4)
    # #calibration
    improvement=calibration(pars1,pars2,pars3,pars4)
    plt.hist(amplitudes1,bins=binsa,color="yellow")
    plt.hist(amplitudes2,bins=binsa,color='green')
    x1=np.linspace(centers[cut1],centers[-1],200)
    x2=np.linspace(centers[0],centers[cut1],200)
    x3=np.linspace(centers[cut2],centers[-1],200)
    x4=np.linspace(centers[0],centers[cut2],200)
    plt.plot(x1,gauss(x1,*pars1),color='black')
    plt.plot(x2,gauss(x2,*pars2),color='blue')
    plt.plot(x3,gauss(x3,*pars3),color='red')
    plt.plot(x4,gauss(x4,*pars4),color='magenta')
    return improvement