import numpy as np
import scipy.io
import threading
import thread
import time
import pyaudio

signal = 0
lock = threading.Lock()
filter_bank = scipy.io.loadmat('filter_bank.mat')
filter_bank = filter_bank['mel_filters']
hamming = np.hamming(256)

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
    while(True):
        lock.acquire(1)

        if(len(signal) != 256):
            w = np.multiply(signal, hamming)
            fft = np.abs(np.fft.rfft(w))
            mel_vals = np.dot(filter_bank, np.transpose(fft))
            print("Processing")

        lock.release()

def generate_signal():
    p = pyaudio.PyAudio()
    stream = p.open(
                    format = pyaudio.paInt16,
                    channels = 1,
                    rate = 8000,
                    input_device_index = 1,
                    input = True)

    global signal

    while(True):
        new_samples = stream.read(256)
        lock.acquire(1)
        signal = np.fromstring(new_samples)
        print("Sampling")
        lock.release()

    
thread1 = BeatThread(generate_signal)
thread2 = BeatThread(process)

thread1.start()
thread2.start()
    
    



