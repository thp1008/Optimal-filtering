import files
import numpy as np
import matplotlib.pyplot as plt
from functions import find_amp,amp_matrix
from scipy.optimize import curve_fit
from scipy.stats import norm
f=files.LJHFile("20200224_112412_chan1.ljh")
amplitudes=np.zeros(0)
amplitudes1=np.zeros(0)
pulsemod=np.load('pulsemodel_chan1.npy')
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
    if abs(2200-amp)<=40:
        amplitudes=np.append(amplitudes,amp)
        amplitudes1=np.append(amplitudes1,amp1)
counts1,bins1=np.histogram(amplitudes,bins=100)
counts2,bins2=np.histogram(amplitudes1,bins=bins1)
centers=(0.5*(bins1[1:]+bins1[:-1]))
range1=np.where((2180<centers)&(centers<2205))[0]
range2=np.where(2205<=centers)[0]

def gauss(x,mu,sig,scl):
    return scl*norm.pdf(x, loc=mu, scale=sig)
pars1,cov1=curve_fit(gauss,centers[range1[0]:range1[-1]], counts1[range1[0]:range1[-1]], p0=[2190,3,60])
pars2,cov2=curve_fit(gauss, centers[range2[0]:range2[-1]], counts1[range2[0]:range2[-1]], p0=[2230,3,80])
pars3,cov3=curve_fit(gauss, centers[range1[0]:range1[-1]], counts2[range1[0]:range1[-1]], p0=[2190,3,60])
pars4,cov4=curve_fit(gauss, centers[range2[0]:range2[-1]], counts2[range2[0]:range2[-1]], p0=[2230,3,80])
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
plt.savefig('figures/dist_pulseamps_at2200.pdf')
#each nth trace is called by trace=f.read_trace(n)
#each trace has a len of 2048
#average amp is 23372.1