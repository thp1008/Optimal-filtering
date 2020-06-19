#plotting stds to increasing noice and the number of samples
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from functions import find_blip,noisemaker,generate_data

# stdn changing with increasing noise
interval1=np.linspace(1,10,50)
deviations1=np.zeros([len(interval1)])
errors1=np.zeros([len(interval1)])
pulse_model=np.load('pulsemodel.npy')
for i in range(len(interval1)):
    print(i)
    n_samples=500
    blip=find_blip(interval1[i],pulse_model)
    n,bins=np.histogram(blip,bins=20,density=True)
    centers=(0.5*(bins[1:]+bins[:-1]))
    pars,var=curve_fit(lambda x, mu, sig : norm.pdf(x, loc=mu, scale=sig), centers, n, p0=[0,1])
    deviations1[i]=pars[1]
    errors1[i]=np.sqrt(var[1,1])
f1=plt.figure()
plt.errorbar(interval1,deviations1,errors1,color='r')
plt.xlabel("noise level")
plt.ylabel("standard deviation")
plt.savefig("figures/std_v_noiselevel.pdf")



# stdn changing with increasing n samples
n_samples=np.linspace(50,1000,100,dtype=int)
deviations2=np.zeros([len(n_samples)])
errors2=np.zeros([len(n_samples)])
for i,n in enumerate(n_samples):
    print(i)
    x_data,y_data=generate_data(n)
    blip=find_blip(1,y_data)
    n,bins=np.histogram(blip,bins=20,density=True)
    centers=(0.5*(bins[1:]+bins[:-1]))
    pars,var=curve_fit(lambda x, mu, sig : norm.pdf(x, loc=mu, scale=sig), centers, n, p0=[0,1])
    deviations2[i]=pars[1]
    

f2,a2=plt.subplots()
a2.errorbar(n_samples,deviations2,errors2,color='r')
a2.set_xlabel("n_samples")
a2.set_ylabel("standard deviation")
f2.savefig("figures/std_v_nsamples.pdf")
f2.show()