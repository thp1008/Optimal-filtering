import files
import numpy as np
import matplotlib.pyplot as plt
f=files.LJHFile("20200224_112412_chan1.ljh")
amplitudes=np.zeros(f.nPulses)
for i in range(len(amplitudes)):
    tr=f.read_trace(i)
    ave=np.average(tr[0:500])
    amplitudes[i]=np.amax(tr)-ave
counts,bins=np.histogram(amplitudes,bins=100,range=(0,8000))
plt.hist(amplitudes,bins=100,range=(0,8000),color="r")
plt.show()
plt.savefig('figures/dist_pulseamps.pdf')
#each nth trace is called by trace=f.read_trace(n)
#each trace has a len of 2048
#average amp is 23372.1