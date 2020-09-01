import os
import files
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functions import amp_matrix,find_amp,gauss,find_spike,model_corr,calibration

            
def improv_graph(f):
    spike,spikewidth=find_spike(f)
    pulsemodel,corr_matrix,amplitudes1=model_corr(f,spike,spikewidth)
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
    countsa,binsa=np.histogram(amplitudes1,bins=int(1.8*spikewidth))
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
    plt.xlabel('adc')
    plt.ylabel('count')
    print(improvement)
count=0
for filename in os.listdir('Data'):
    if filename.endswith(('.ljh')):
        f=files.LJHFile('Data/'+filename)
        
        count+=1
        if count==6:
            improv_graph(f)
        elif count>6    :
            break
    