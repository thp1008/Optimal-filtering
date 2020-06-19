#effects of time jitter on amplitude estimation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from functions import toysim,noisemaker,generate_data,amp_matrix,find_amp,generate_pulsemod
n_samples=500
xdata1,ydata1=generate_data(n_samples)
xdata2,ydata2=generate_data(n_samples,peak=4+(3.5)*(10/n_samples))

pulsemod=generate_pulsemod(ydata1)
# pulsemod2=generate_pulsemod(noisemaker(ydata2))

block2=amp_matrix(pulsemod)
# block2b=amp_matrix(pulsemod2)

# measured1=noisemaker(ydata1)
# measured2=noisemaker(ydata2)

# amp1=find_amp(measured1,block2)
# amp2=find_amp(measured2,block2)
# print(amp1,amp2)
# blip1=np.zeros(1000)
# blip2=np.zeros(1000)
# for i in range(1000):
#     ydata=noisemaker(ydata2)
#     amp=find_amp(ydata,block2)
#     blip1[i]=1-amp
# mu,sigma=norm.fit(blip1)
# n,bins,patches=plt.hist(blip1,bins=30)
# binwidth=bins[1]-bins[0]
# x=np.linspace(bins[0],bins[-1],1000)
# y = norm.pdf(x,mu,sigma)*len(blip1)*binwidth
# l = plt.plot(x, y, 'r', linewidth=2)
# plt.show()

shift=np.linspace(0,4,8)
deviations=np.zeros([len(shift)])
errors=np.zeros([len(shift)])
for i,n in enumerate(shift):
    print(i)
    xdata,ydata=generate_data(n_samples,peak=4+n*(10/n_samples))
    blip=np.zeros(1000)
    for z in range(1000):
        ydata1=noisemaker(ydata)
        amp=find_amp(ydata1,block2)
        blip[z]=1-amp
    n,bins=np.histogram(blip,bins=20,density=True)
    centers=(0.5*(bins[1:]+bins[:-1]))
    pars,var=curve_fit(lambda x, mu, sig : norm.pdf(x, loc=mu, scale=sig), centers, n, p0=[0,1])
    deviations[i]=pars[0]
    errors[i]=np.sqrt(var[0,0])
f1=plt.figure()
plt.errorbar(shift,deviations,errors,color='r')
plt.xlabel("time offset")
plt.ylabel("mean")
plt.savefig("figures/mean_v_timeshift_width.pdf")
# fig,(ax1,ax2)=plt.subplots(2)
# ax1.plot(xdata1,ydata1)
# ax2.plot(xdata2,ydata2,"r")