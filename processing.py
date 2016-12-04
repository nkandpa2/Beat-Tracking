import numpy as np
import scipy.io
import threading
import thread
import time

signal = 0
state = 0
lock = threading.Lock()
filter_bank = scipy.io.loadmat('filter_bank.mat')
filter_bank = filter_bank['mel_filters']

class BeatThread(threading.Thread):
    def __init__(self, threadID, function_num):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.function_num = function_num

    def run(self):
        if(self.function_num == 0):
            generate_signal()
        else:
            process()


def process():
    print("Thread 2 Started")
    while(True):
        global state
        if(state == 1):
            lock.acquire(1)
            global signal
            global filter_bank
            w = np.multiply(signal, np.hamming(len(signal)))
            fft = np.abs(np.fft.rfft(w))
            mel_vals = np.dot(filter_bank, np.transpose(fft))
            print("Processing")
            lock.release()
            state = 0

def generate_signal():
    print("Thread 1 Started")
    while(True):
        global state
        if(state == 0):
            lock.acquire(1)
            global signal
            signal = np.random.rand(1,256)
            print("Sampling")
            lock.release()
            state = 1

    
thread1 = BeatThread(0, 0)
thread2 = BeatThread(1, 1)

thread1.start()
time.sleep(1)
thread2.start()
    
    



