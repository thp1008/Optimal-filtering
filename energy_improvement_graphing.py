#finding resolution improvement for all detectors
# step1:
import os
import files
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functions import amp_matrix,find_amp,gauss,find_spike,model_corr,calibration
resolution_diff=[]
counts=0
for filename in os.listdir('Data'):
    if filename.endswith('chan117.ljh') or filename.endswith('chan119.ljh') or filename.endswith('chan121.ljh') or filename.endswith('chan135.ljh') or filename.endswith('chan21.ljh') or filename.endswith('chan23.ljh'):
        continue
    if filename.endswith('chan25.ljh'):
        break
    if filename.endswith(('.ljh')):
        print(filename)
        #finding the 40.9 and 41.5 keV spike
        #spike width is 50
        f=files.LJHFile('Data/'+filename)
        spike=find_spike(f)
        spikewidth=50
        
        #making a pulsemodel and finding the least mean square
        pulsemodel,corr_matrix,amplitudes1=model_corr(f,spike)
        
        #optimal filtering
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
                amplitudes2=np.append(amplitudes1,amp1)
        countsa,binsa=np.histogram(amplitudes1,bins=100)
        i_amax=np.argmax(countsa)
        countsb,binsb=np.histogram(amplitudes2,bins=binsa)
        centers=(0.5*(binsa[1:]+binsa[:-1]))
        range1=np.arange(len(centers)//2)
        range2=np.arange(len(centers)//2,len(centers))
        pars1,cov1=curve_fit(gauss, centers[range1], countsa[range1], p0=[centers[np.argmax(countsa[range1])],4,np.amax(countsa[range1])])
        pars2,cov2=curve_fit(gauss, centers[range2], countsa[range2], p0=[centers[np.argmax(countsa[range2])],4,np.amax(countsa[range2])])
        pars3,cov3=curve_fit(gauss, centers[range1], countsb[range1], p0=[centers[np.argmax(countsb[range1])],4,np.amax(countsb[range1])])
        pars4,cov4=curve_fit(gauss, centers[range2], countsb[range2], p0=[centers[np.argmax(countsb[range2])],4,np.amax(countsb[range2])])
        
        #calibration
        improvement=calibration(pars1,pars2,pars3,pars4)
        resolution_diff.append(improvement)
#plotting
np.histogram(resolution_diff,bins=10)
plt.hist(resolution_diff,bins=10)
plt.xlabel('resolution improvement')
plt.ylabel('count')

        
        
        
        
        
        