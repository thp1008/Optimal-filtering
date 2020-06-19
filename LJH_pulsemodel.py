import files
import numpy as np
import matplotlib.pyplot as plt
f=files.LJHFile("20200224_112412_chan1.ljh")
amplitudes=np.zeros(f.nPulses)

#generate pulse model based on peak at 2200
traces=np.zeros(2048)
for i in range(len(amplitudes)):
    tr=f.read_trace(i)
    ave=np.average(tr[0:500])
    amp=np.amax(tr)-ave
    if abs(2200-amp)<=40:
        traces=traces+(tr-ave)
pulsemod=traces/968
x=np.arange(2048)
plt.plot(x,pulsemod)
plt.show()