import math
import numpy as np
import matplotlib.pyplot as plt



label1="Nice Job"



def toysim(x):
    N1=10.0
    N2=1.0
    if x<=7.3:
        y=N1*math.exp(-((x-6)/3)**2)
    else:
        y=N2*math.exp(-1/3*(x-13.66))
    return y



def noisemaker(dataset):
    setsize=np.size(dataset)
    newset=np.zeros(setsize)
    for i in range(setsize):
        newset[i]=np.random.normal(dataset[i],1)
        i=i+1
    return newset

x_data=np.linspace(0,20,1000)
y_data=np.zeros(1000)



for i in range(len(x_data)):
    y_data[i]=toysim(x_data[i])
    i=i+1

z_data=noisemaker(y_data)

fig = plt.figure(1)
plt.plot(x_data,y_data,color='blue',label=label1)
plt.plot(x_data,z_data,color='yellow')