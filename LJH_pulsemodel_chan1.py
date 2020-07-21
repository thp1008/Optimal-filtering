import files
import numpy as np
import matplotlib.pyplot as plt
f=files.LJHFile("20200224_112412_chan1.ljh")
npulses=np.zeros(f.nPulses)
#generate pulse model based on peak at 2200
traces=np.zeros(1285)
count=0
traces1=np.zeros(1285)
count1=0
traces2=np.zeros(1285)
count2=0
for i in range(len(npulses)):
    tr=f.read_trace(i)
    ave=np.average(tr[0:500])
    amp=np.amax(tr)-ave
    tr=tr[515:1800]
    if abs(2200-amp)<=40:
        traces=traces+(tr-ave)
        count+=1
    elif abs(5400-amp)<=40:
        traces1=traces1+(tr-ave)
        count1+=1
    elif abs(2520-amp)<=40:
        traces2=traces2+(tr-ave)
        count2+=1
pulsemod=traces/count
pulsemod1=traces1/count1
pulsemod2=traces2/count2
np.save('pulsemodel.npy',pulsemod)
np.save('pulsemodel5400.npy',pulsemod1)
x=np.arange(1285)
plt.plot(x,pulsemod1)
plt.show()
plt.savefig('figures/pulsemodel.pdf')

# plt.plot(x,pulsemod,label='amp 2200')
# plt.plot(x,pulsemod1,label='amp 5400')       
# plt.plot(x,pulsemod2,label='amp 2520')
# y=plt.subplot()
# z=plt.subplot()
# plt.plot(x[515:],(pulsemod/pulsemod1)[515:],label='amp 5400')
# z.plot(x[515:],(pulsemod/pulsemod2)[515:],label='amp 2520')

# plt.xlabel('time samples')
# plt.ylabel('ratio')
# plt.legend()
# plt.show()
# plt.savefig('figures/1mainpulse_div_higherEpulse.pdf')