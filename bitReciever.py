# import sounddevice as sd
# from scipy.io.wavfile import write
import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import pyaudio
import wave


fs = 40000  # Sample rate
seconds = 10 # Duration of recording
'''
def fft_plot(audio, sr):
    n = len(audio)
    T = 1/sr
    yf = scipy.fft.fft(audio)
    xf = np.linspace(0.0, 1/(2*T), n//2)
    
    fig, ax = plt.subplots()
    
    ax.plot(xf,2.0/n * np.abs(yf[:n//2]))
    ax.set_xlim(400,2500)
    plt.grid()
    plt.xlabel("Freq ->")
    plt.ylabel("Mag")
    return plt.show()
'''
p = pyaudio.PyAudio()
con = input('Press y to proceed:')
if con == 'y' or con == 'Y':
    # myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
    
    # sd.wait()  # Wait until recording is finished
    # write('output.wav', fs, myrecording)  # Save as WAV file 
      # Create an interface to PortAudio
     
    print('-----Now Recording-----')
    chunk = 1024      # Each chunk will consist of 1024 samples
    sample_format = pyaudio.paInt16      # 16 bits per sample
    channels = 1     # Number of audio channels
    fs = 44100        # Record at 44100 samples per second
    time_in_seconds = 6
    filename = "output.wav"
    #Open a Stream with the values we just defined
    stream = p.open(format=sample_format,
                    channels = channels,
                    rate = fs,
                    frames_per_buffer = chunk,
                    input = True)
     
    frames = []  # Initialize array to store frames
     
    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * time_in_seconds)):
        data = stream.read(chunk)
        frames.append(data)
     
    # Stop and close the Stream and PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
 
print('-----Finished Recording-----')

    



chunk = 2048

# open up a wave
wf = wave.open('output.wav', 'rb')
swidth = wf.getsampwidth()
RATE = wf.getframerate()
# use a Blackman window
window = np.blackman(chunk)
# open stream
p = pyaudio.PyAudio()
stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = RATE,
                output = True)

# read some data
data = wf.readframes(chunk)
l = []
# play stream and find the frequency of each chunk
while len(data) == chunk*swidth:
    # write data out to the audio stream
    stream.write(data)
    # unpack the data and times by the hamming window
    indata = np.array(wave.struct.unpack("%dh"%(len(data)/swidth),\
                                         data))*window
    # Take the fft and square each value
    fftData=abs(np.fft.rfft(indata))**2
    # find the maximum
    which = fftData[1:].argmax() + 1
    # use quadratic interpolation around the max
    if which != len(fftData)-1:
        y0,y1,y2 = np.log(fftData[which-1:which+2:])
        x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
        # find the frequency and output it
        thefreq = (which+x1)*RATE/chunk
        l.append(thefreq)
        #print("The freq is %f Hz." % (thefreq))
        
    else:
        print('ELSE')
        thefreq = which*RATE/chunk
        l.append(thefreq)
        #print("The freq is %f Hz." % (thefreq))
    # read some more data
    data = wf.readframes(chunk)
if data:
    stream.write(data)
stream.close()
p.terminate()


le = len(l)
i = 0
flag = False
win = 6
suru = False
store = []
while True:
    if not suru and 1900<= l[i] <= 2100:
        suru = True
    if not suru:
        i += 1
    if suru:
        avg = sum(l[i:i+win])//win
        if 1900<= avg <= 2100:
            flag = not flag
        if flag and 800 <= avg <= 2000:
            store.append(avg)
        i += win
        if i > le-win:
            break

out = ''

for each in store:
    if 1900<=each<= 2000:
        out = out+'S'
    elif 1100<= each <= 1400:
        out = out+'1'
    elif 750 <= each <= 950:
        out = out+'0'

print(out)
# file_p = "output.wav"
# samples, sampling_rate = librosa.load(file_p, sr = None, mono = False, offset = 0.0,duration = None)

#fft_plot(samples,sampling_rate)