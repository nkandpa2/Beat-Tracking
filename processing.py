import numpy as np
import scipy.io
import time
import pyaudio

filter_bank = scipy.io.loadmat('filter_bank.mat')
filter_bank = filter_bank['mel_filters']
hamming = np.hamming(256)
def process(in_data, frame_count, time_info, status_flags):
    global hamming, filter_bank
    window = np.fromstring(in_data, 'Float32')
    window = np.multiply(window, hamming)
    fft = np.abs(np.fft.rfft(window))
    mel_vals = np.dot(filter_bank, np.transpose(fft))
    return(None, pyaudio.paContinue)

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    stream = p.open(format = pyaudio.paFloat32,
                    channels = 1,
                    rate = 8000,
                    input_device_index = 1,
                    frames_per_buffer = 256,
                    input = True,
                    stream_callback = process
                    )
    while(stream.is_active()):
        time.sleep(0.1)

    stream.close()
    p.terminate()



