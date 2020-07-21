import files
import numpy as np
import matplotlib.pyplot as plt
from functions import amp_matrix,find_amp
f=files.LJHFile("20200224_112412_chan1.ljh")
amplitudes=np.zeros(f.nPulses)
amplitudes1=np.zeros(f.nPulses)
pulsemod=np.load('pulsemodel_chan1.npy')
block2=amp_matrix(pulsemod)
for i in range(len(amplitudes)):
    tr=f.read_trace(i)
    ave=np.average(tr[0:500])
    tr=tr[515:1800]
    amplitudes[i]=np.amax(tr)-ave
    scale=find_amp(tr,block2)
    amplitudes1[i]=scale*np.amax(pulsemod)
counts,bins=np.histogram(amplitudes,bins=100,range=(0,8000))
# plt.hist(amplitudes,bins=300,range=(0,6000),color="r")
plt.hist(amplitudes1,bins=100,range=(0,8000),color="r")
plt.show()
# plt.savefig('figures/dist_pulseamps.pdf')
#each nth trace is called by trace=f.read_trace(n)
#each trace has a len of 2048
#average amp is 23372.1