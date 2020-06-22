import files
import numpy as np
import matplotlib.pyplot as plt
f=files.LJHFile("20200224_112412_chan1.ljh")
npulses=np.zeros(f.nPulses)
#generate pulse model based on peak at 2200
traces=np.zeros(2048)
traces1=np.zeros(2048)
traces2=np.zeros(2048)
for i in range(len(npulses)):
    tr=f.read_trace(i)
    ave=np.average(tr[0:500])
    amp=np.amax(tr)-ave
    if abs(2200-amp)<=40:
        traces=traces+(tr-ave)
    elif abs(5400-amp)<=40:
        traces1=traces1+(tr-ave)
    elif abs(2520-amp)<=40:
        traces2=traces2+(tr-ave)
pulsemod=traces/968
pulsemod1=traces1/968
pulsemod2=traces2/968
# np.save('pulsemodel.npy',pulsemod)
x=np.arange(2048)
# plt.plot(x,pulsemod)
# plt.show()
# plt.savefig('figures/pulsemodel.pdf')

# plt.plot(x,pulsemod)
# plt.plot(x,pulsemod1)       
# plt.plot(x,pulsemod2)
y=plt.subplot()
z=plt.subplot()
plt.plot(x,(pulsemod/pulsemod1))
z.plot(x,(pulsemod/pulsemod2))
plt.show()
plt.savefig('figures/1mainpulse_div_higherEpulse.pdf')