import numpy as np
import scipy.io
import scipy.signal
import time
import pyaudio

filter_bank = scipy.io.loadmat('filter_bank.mat')
filter_bank = filter_bank['mel_filters']

hamming = np.hamming(256)

gausswin = scipy.signal.gaussian(5, std=2)

onset_env_conv = np.zeros((35*250))
onset_env = np.zeros((35*250))
cum_score = np.zeros((35*250))

signal = np.empty((0))
last_mel = np.empty((0,40))


curr_window = np.empty((0))

frame_num = 1
tempo = 187 #bpm
#tempo_samples = np.round(250.0*60/tempo)
tempo_samples = -1
index = 0
alpha = -1 

threshold = 10
num_increasing = 0
num_decreasing = 0
peak_detected = False
num_beats = 0
beats = np.empty((0))

r = np.arange(np.round(tempo_samples/2), tempo_samples*2, 1)
prev_beat_window = -400*np.power((np.log(r/(float)(tempo_samples))), 2)
prev_beat_window = np.fliplr([prev_beat_window])[0]

def autocorrelate(onset_env, max_lag):
    samples_per_beat = 1.0 / (120.0/60.0/250.0) #Defaults to 120 bpm or 125 samples per beat
    acr = np.zeros(max_lag)
    onset_lag = onset_env

    for i in range(0,max_lag):
        onset_lag = np.append(np.zeros(i), onset_env[0:len(onset_env) - i]) 
        acr[i] = np.correlate(onset_env, onset_lag)
    
    r = np.arange(max_lag)
    window = np.exp(-0.5*np.power(np.log2(r/samples_per_beat), 2))
    win_acr = np.multiply(window, acr)
    z_pad = np.append(np.append([0],win_acr),[0])
    tps2 = (win_acr[0:np.ceil(len(win_acr)/2.0)] + 
           0.5*z_pad[1:len(win_acr) + 1:2] +
           0.25*z_pad[0:len(win_acr):2] +
           0.25*z_pad[2:len(win_acr) + 2:2])

    tps3 = (win_acr[0:np.ceil(len(win_acr)/3.0)] + 
           0.33*z_pad[1:len(win_acr) + 1:3] +
           0.33*z_pad[0:len(win_acr):3] +
           0.33*z_pad[2:len(win_acr) + 2:3])
    if(np.max(tps2) > np.max(tps3)):
        startpd = np.argmax(tps2)
        startpd2 = startpd*2
    else:
        startpd = np.argmax(tps3)
        startpd2 = startpd*3


    if(win_acr[startpd] > win_acr[startpd2]):
        return startpd
    else:
        return startpd2



def process(in_data, frame_count, time_info, status_flags):
    global hamming, filter_bank, last_mel, onset_env, frame_num, curr_window, gausswin, index, signal, alpha, tightness, threshold, num_increasing, num_decreasing, peak_detected, num_beats, beats, tempo_samples, prev_beat_window
    t3 = time.time()

    if(frame_num < 8):
        window = np.fromstring(in_data, 'Int16')
        signal = np.append(signal, window)
        curr_window = np.append(curr_window, window)

    else:
        window = np.fromstring(in_data, 'Int16')
        curr_window = np.append(curr_window[len(curr_window) - 224:len(curr_window)], window)

        h = np.multiply(curr_window, hamming)
        fft = np.abs(np.fft.rfft(h))
        mel_vals = np.dot(filter_bank, np.transpose(fft))
        mel_vals = 20*np.log10(mel_vals)

        if(frame_num == 8):
            last_mel = mel_vals

        else:
            rectified_diff = np.maximum(np.zeros((1,40)), (mel_vals - last_mel))
            next_onset = np.sum(rectified_diff)
            onset_env_conv[index:index+len(gausswin)] += gausswin*next_onset
            onset_env[index] += next_onset
            #if(index < 2*tempo_samples):
            if(index < 500):
                cum_score[index] = onset_env_conv[index]
            else:
                if(alpha == -1):
                    alpha = np.std(onset_env_conv[0:index-1])

                if(tempo_samples == -1):
                    tempo_samples = autocorrelate(onset_env_conv[0:1000], 500)
                    #tempo_samples = 81
                    r = np.arange(np.round(tempo_samples/2), tempo_samples*2, 1)
                    prev_beat_window = -400*np.power((np.log(r/(float)(tempo_samples))), 2)
                    prev_beat_window = np.fliplr([prev_beat_window])[0]

                    print("Tempo estimate: " + str(250.0 / tempo_samples * 60.0))
                    print("Tempo Samples: " + str(tempo_samples))

                prev_beat = alpha*prev_beat_window + cum_score[index - (int)(2*tempo_samples) : index - (int)(np.round(tempo_samples/2))]
                cum_score[index] = onset_env_conv[index] + np.max(prev_beat)

                
                if(not peak_detected): 
                    if(cum_score[index] > cum_score[index - 1]):
                        num_increasing += 1
                    elif(cum_score[index] < cum_score[index - 1] and num_increasing > threshold):
                        num_increasing = 0
                        peak_detected = True
                else:
                    if(num_decreasing > threshold):
                        print("BEAT: " + (str)(num_beats) + "\n\n\n\n")
                        num_beats += 1
                        peak_detected = False
                        beats = np.append(beats, index)
                        num_decreasing = 0
                    elif(cum_score[index] < cum_score[index - 1]):
                        num_decreasing += 1
                    elif(cum_score[index] > cum_score[index - 1]):
                        num_decreasing = 0
                        peak_detected = True


            last_mel = mel_vals
            index += 1
            
    frame_num += 1
    return(None, pyaudio.paContinue)

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    stream = p.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = 8000,
                    input_device_index = 1,
                    frames_per_buffer = 32,
                    input = True,
                    stream_callback = process
                    )
    t1 = time.time()
    t2 = time.time()
    while(t2 - t1 < 15):
        time.sleep(0.1)
        t2 = time.time()
    stream.close()
    scipy.io.savemat('vars.mat', mdict = {'onset_env_conv':onset_env_conv, 'onset_env':onset_env, 'cum_score':cum_score, 'beats':beats})
