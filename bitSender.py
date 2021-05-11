import sounddevice as sd
import numpy as np
from math import pi
import time
import matplotlib.pyplot as plt



msg_bits = input('Enter binary data to transmit:')
len_msg = len(msg_bits)
sampling_freq = 41000


t = np.arange(0,0.3,1/sampling_freq)


ssBit = np.sin(2*pi*(2000)*t)
on = np.sin(2*pi*(1250)*t)
off = np.sin(2*pi*(880)*t)

start = ssBit


for each in msg_bits:
    if each == '1':
        start = np.concatenate((start,on))
    else:
        start = np.concatenate((start,off))
start = np.concatenate((start,ssBit))

plt.plot(start)
plt.show()
time.sleep(1)

sd.play(start, 41000)
sd.wait()
