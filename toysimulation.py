import math
import numpy as np
import matplotlib.pyplot as plt



label1="Nice Job"



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



def noisemaker(dataset):
    setsize=np.size(dataset)
    newset=np.zeros(setsize)
    for i in range(setsize):
        newset[i]=np.random.normal(dataset[i],1)
        i=i+1
    return newset
n=500
x_data=np.linspace(0,15,n)
y_data=np.zeros(n)



for i in range(len(x_data)):
    y_data[i]=toysim(x_data[i])
    i=i+1


measured_data=noisemaker(.2*y_data)


gen=noisemaker(y_data)
for i in range(999):
    noise=noisemaker(y_data)
    gen=gen+noise
pulse_model=gen/1001
M=np.ones((n,2))
M[:,0]=pulse_model
R=np.diag([1]*n)





def find_amp(dataset):
    MT=M.transpose()
    invR=np.linalg.inv(R)
    block1=np.linalg.inv(np.dot(MT,np.dot(invR,M)))
    block2=np.dot(block1,np.dot(MT,invR))
    p_hat=np.dot(block2,dataset)
    amplitude=p_hat[0]
    return amplitude
amp=find_amp(measured_data)
print(amp)

fig = plt.figure(1)
# plt.plot(x_data,pulse_model,color='red')


plt.plot(x_data,measured_data,color='yellow')
plt.plot(x_data,amp*pulse_model,color='blue')



