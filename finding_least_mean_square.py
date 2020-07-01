import files
import numpy as np
f=files.LJHFile("20200224_112412_chan1.ljh")
npulses=np.zeros(f.nPulses)
x=0
xsq=0
count=0
for i in range(len(npulses)):
    tr=f.read_trace(i)
    ave=np.average(tr[0:500])
    amp=np.amax(tr)-ave
    tr=tr[515:1800]
    if abs(2200-amp)<=40:
        x=x+amp
        xsq+=amp**2
        count+=1
var=(xsq/count-(x/count)**2)**(1/2)
print(var)