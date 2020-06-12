import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from functions import find_blip



# def toysim(x):
#     N1=10.0
#     mu=4
#     sigma=5
#     tau=.5
#     t_naught=(sigma**2)/(2*tau)
#     N2=N1*math.exp(-((t_naught-mu)/sigma)**2+(t_naught/tau))
#     if x<=t_naught:
#         y=N1*math.exp(-((x-mu)/sigma)**2)
#     else:
#         y=N2*math.exp(-x/tau)
#     return y

# def generate_data(n_samples):
#     x_data=np.linspace(0,15,n_samples)
#     y_data=np.zeros(n_samples)
#     for i in range(len(x_data)):
#         y_data[i]=toysim(x_data[i])
#         i=i+1
#     return x_data,y_data

# def noisemaker(dataset,intensity):
#     setsize=np.size(dataset)
#     newset=np.zeros(setsize)
#     for i in range(setsize):
#         newset[i]=np.random.normal(dataset[i],intensity)
#         i=i+1
#     return newset







# measured_data=noisemaker(y_data,1)


# gen=noisemaker(y_data,1)
# for i in range(999):
#     noise=noisemaker(y_data,1)
#     gen=gen+noise
# pulse_model=(gen/1001)

# M=np.ones((n_samples,2))
# M[:,0]=pulse_model
# R=np.diag([1]*n_samples)





# def find_amp(dataset,n_samples):
#     pulse_model=np.load("pulsemodel.npy")
#     M=np.ones((n_samples,2))
#     M[:,0]=pulse_model
#     MT=M.transpose()
#     R=np.diag([1]*n_samples)
#     invR=np.linalg.inv(R)
#     block1=np.linalg.inv(np.dot(MT,np.dot(invR,M)))
#     block2=np.dot(block1,np.dot(MT,invR))
#     p_hat=np.dot(block2,dataset)
#     amplitude=p_hat[0]
#     return amplitude
# amp=find_amp(measured_data)
# print(amp)

# fig = plt.figure(1)
# plt.plot(x_data,pulse_model,color='red')


# plt.plot(x_data,measured_data,color='yellow')
# plt.plot(x_data,amp*pulse_model,color='blue')


# def find_blip(intensity,n_samples):
#     blip=np.zeros(100)
#     for i in range(len(blip)):
#         x_data,y_data=generate_data(n_samples)
#         measured_data=noisemaker(y_data,intensity)
#         blip[i]=(1-find_amp(measured_data,n_samples))
#     return blip
             
           

# mu,sigma=norm.fit(blip)
# n,bins,patches=plt.hist(blip,bins=30)
# binwidth=bins[1]-bins[0]
# x=np.linspace(bins[0],bins[-1],1000)
# y = norm.pdf(x,mu,sigma)*len(blip)*binwidth
# l = plt.plot(x, y, 'r', linewidth=2)
# plt.show()
# bins=bins[:-1]
# bins=bins+.5*binswidth





# stdn changing with increasing noise
interval1=np.linspace(1,10,5)
deviations1=np.zeros([len(interval1)])
errors1=np.zeros([len(interval1)])
for i in range(len(interval1)):
    print(i)
    n_samples=500
    blip=find_blip(interval1[i],n_samples)
    # mu,sigma=norm.fit(blip)
    # deviations[i]=sigma
    n,bins=np.histogram(blip,bins=20,density=True)
    centers=(0.5*(bins[1:]+bins[:-1]))
    pars,var=curve_fit(lambda x, mu, sig : norm.pdf(x, loc=mu, scale=sig), centers, n, p0=[0,1])
    deviations1[i]=pars[1]
    errors1[i]=np.sqrt(var[1,1])
f1=plt.figure()
plt.errorbar(interval1,deviations1,errors1,color='r')
plt.show()





# stdn changing with increasing n samples
# n_samples=np.linspace(50,1000,500)

# f2=plt.figure()
# plt.show()