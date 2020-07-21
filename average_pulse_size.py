import os
import files
avepulsesize=[]
for filename in os.listdir('Data'):
    if filename.endswith(('.ljh')):
        size=(os.path.getsize('Data/'+filename))
        f=files.LJHFile('Data/'+filename)
        num=f.nPulses
        avepulsesize.append(size/num)
avesize=sum(avepulsesize)/len(avepulsesize)
print(avesize,'bytes')