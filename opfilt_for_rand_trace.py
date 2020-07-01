import files
import numpy as np
import matplotlib.pyplot as plt
from functions import amp_matrix, find_amp
from random import randint
f=files.LJHFile("20200224_112412_chan1.ljh")
#optimal filtering for random pulses

x=np.arange(1285)
tr=f.read_trace(randint(0,f.nPulses))
trace=tr[515:1800]-np.average(tr[0:500])

pulsemod=np.load('pulsemodel.npy')
block2=amp_matrix(pulsemod)

scale=find_amp(trace,block2)
amplitude=scale*np.amax(pulsemod)
plt.plot(x,trace,'b')
plt.plot(x,pulsemod*scale,'r')
plt.plot([0,1285],[amplitude,amplitude])
plt.xlabel('time samples')
plt.ylabel('adc')
plt.savefig('figures/findamp_randpulse.pdf')