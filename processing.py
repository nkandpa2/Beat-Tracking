import numpy as np
import scipy.io
import threading
import thread
import time
import pyaudio

class BeatThread(threading.Thread):
    def __init__(self, function):
        threading.Thread.__init__(self)
        self.function = function

    def run(self):
        self.function()

def process():
    global filter_bank
    global signal
    global hamming
    global state

    while(True):
        if(state != 1):
            continue

        lock.acquire(1)
        if(len(signal) == 256):
            w = np.multiply(signal, hamming)
            fft = np.abs(np.fft.rfft(w))
            mel_vals = np.dot(filter_bank, np.transpose(fft))
            print(mel_vals)
            print("Processing")

        lock.release()
        state = 0

def generate_signal():
    p = pyaudio.PyAudio()
    stream = p.open(
                    format = pyaudio.paFloat32,
                    channels = 1,
                    rate = 8000,
                    input_device_index = 1,
                    frames_per_buffer = 256,
                    input = True)

    global signal
    global state
    while(True):
        if(state != 0):
            continue
        new_samples = stream.read(256)
        t1 = time.time()
        lock.acquire(1)
        signal = np.fromstring(new_samples, 'Float32')
        print("Sampling Audio")
        state = 1
        lock.release()

signal = [0]
lock = threading.Lock()
filter_bank = scipy.io.loadmat('filter_bank.mat')
filter_bank = filter_bank['mel_filters']
hamming = np.hamming(256)
state = 0

if __name__ == "__main__":   
    thread1 = BeatThread(generate_signal)
    thread2 = BeatThread(process)

    thread1.start()
    thread2.start()



