import files
import numpy as np
import matplotlib.pyplot as plt
from functions import find_amp,amp_matrix
from scipy.optimize import curve_fit
from scipy.stats import norm
f=files.LJHFile("20200224_112412_chan15.ljh")
amplitudes=np.zeros(0)
amplitudes1=np.zeros(0)
pulsemod=np.load('pulsemodel_chan15.npy')
block2=amp_matrix(pulsemod)
# def gauss(x,mu,sigma,A):
#     return A*np.exp(-(x-mu)**2/2/sigma**2)
# def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2,offset):
#     return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+offset
for i in range(f.nPulses):
    tr=f.read_trace(i)
    ave=np.average(tr[0:500])
    amp=np.amax(tr)-ave
    tr=tr-ave
    tr=tr[515:1800]
    scale=find_amp(tr,block2)
    amp1=scale*np.amax(pulsemod)
    if amp>=1625 and amp<=1675:
        amplitudes=np.append(amplitudes,amp)
        amplitudes1=np.append(amplitudes1,amp1)
counts1,bins1=np.histogram(amplitudes,bins=100)
counts2,bins2=np.histogram(amplitudes1,bins=bins1)
centers=(0.5*(bins1[1:]+bins1[:-1]))
range1=np.where((1625<centers)&(centers<1650))[0]
range2=np.where(1650<=centers)[0]

def gauss(x,mu,sig,scl):
    return scl*norm.pdf(x, loc=mu, scale=sig)
pars1,cov1=curve_fit(gauss,centers[range1[0]:range1[-1]], counts1[range1[0]:range1[-1]], p0=[1637,5,20])
pars2,cov2=curve_fit(gauss, centers[range2[0]:range2[-1]], counts1[range2[0]:range2[-1]], p0=[1665,5,30])
pars3,cov3=curve_fit(gauss, centers[range1[0]:range1[-1]], counts2[range1[0]:range1[-1]], p0=[1635,5,40])
pars4,cov4=curve_fit(gauss, centers[range2[0]:range2[-1]], counts2[range2[0]:range2[-1]], p0=[1660,5,50])
plt.hist(amplitudes,bins=bins1,color="yellow")
plt.hist(amplitudes1,bins=bins2,color='green')
x1=np.linspace(centers[range1[0]],centers[range1[-1]],200)
x2=np.linspace(centers[range2[0]],centers[range2[-1]],200)
plt.plot(x1,gauss(x1,*pars1),color='black')
plt.plot(x2,gauss(x2,*pars2),color='black')
plt.plot(x1,gauss(x1,*pars3),color='red')
plt.plot(x2,gauss(x2,*pars4),color='red')

# plt.plot(centers1,bimodal(centers1,*expected),color='purple')
# plt.plot(centers2,bimodal(centers2,*expected),color='black')

plt.show()
plt.ylabel('frequency')
plt.xlabel('energy level')
# plt.savefig('figures/dist_pulseamps_at2200.pdf')
#each nth trace is called by trace=f.read_trace(n)
#each trace has a len of 2048
#average amp is 23372.1

#calibration
slope1=-.6403/(pars1[0]-pars2[0])
slope2=-.6403/(pars3[0]-pars4[0])
offset1=41.5427-pars2[0]*slope1
offset2=41.5427-pars4[0]*slope2
energyres1=pars1[1]*slope1
energyres2=pars3[1]*slope2
energyres3=pars2[1]*slope1
energyres4=pars4[1]*slope2

improv=energyres2/energyres1
print(improv)
print(slope1,slope2,offset1,offset2,energyres1,energyres2,energyres3,energyres4)