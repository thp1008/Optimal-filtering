import files
import numpy as np
import matplotlib.pyplot as plt
from functions import amp_matrix, find_amp
from random import randint
f=files.LJHFile("20200224_112412_chan1.ljh")
#optimal filtering for random pulses

x=np.arange(2048)
tr=f.read_trace(randint(0,f.nPulses))
trace=tr-np.average(tr[0:500])

pulsemod=np.load('pulsemodel.npy')
block2=amp_matrix(pulsemod)

amplitude=find_amp(trace,block2)*np.amax(pulsemod)

plt.plot(x,trace)
plt.plot([0,2048],[amplitude,amplitude])
plt.savefig('figures/findamp_randpulse.pdf')