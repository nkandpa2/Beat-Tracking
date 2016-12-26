import numpy as np
import scipy.io
import scipy.signal
import time
import pyaudio

class BeatTracker:

############################################################################################################
    #Constructor that initializes all of the variables necessary for algorithm
    def __init__(self):
        #Mel Filter Bank
        self.filter_bank = scipy.io.loadmat('filter_bank.mat')['mel_filters']

        #Hamming Window 
        self.hamming = np.hamming(256)

        #Gaussian for smoothing of onset envelope
        self.gaussian = scipy.signal.gaussian(5, std=2)

        #Onset Envelope
        self.onset_env = np.zeros((35*250))

        #Cumulative Beat Score
        self.cum_score = np.zeros((35*250))

        #Mel values from last window
        self.last_mel = np.zeros((0,40))

        #Audio signal from current 4 ms window
        self.curr_window = np.zeros((0))

        #Current frame number (frames are 4ms long)
        self.frame_num = 1

        #Global tempo estimate given in number of samples between beats
        self.tempo_samples = -1

        #Index being written to in onset envelope and cumulative score
        self.index = 0

        #Parameter for how highly the scoring function weights the previous beat
        self.alpha = -1

        #Number of decreasing values in cum_score that need to be seen before a peak is recognized
        self.threshold = 5

        #Flag for when we see a peak in the cumulative score
        self.peak_detected = False

        #Counter for number of consecutive decreasing time steps in cumulative score
        self.num_decreasing = 0

        #Counter for number of consecutive increasing time steps in cumulative score
        self.num_increasing = 0

        #Previous cumulative score peak value
        self.prev_peak = 0

        #Number of beats seen
        self.num_beats = 0

        #Keeps indices where beats were found
        self.beats = np.zeros((0))

        #Previous beat weighting window
        self.prev_beat_window = np.zeros((0))
    
    
############################################################################################################

    #Adds 4 ms of new audio to last 28 ms of previous window to give past 32 ms of audio
    def create_window(self, new_samples):
        if(self.frame_num < 8):
            self.curr_window = np.append(self.curr_window, new_samples)

        else:
            self.curr_window = np.append(self.curr_window[len(self.curr_window)-224:len(self.curr_window)], new_samples)

        self.frame_num += 1
    
    
############################################################################################################

    #Calculates the next onset value based on the current window and adds this onset value to the onset envelope
    def calc_onset(self):
        if(self.frame_num > 8):
            h = np.multiply(self.curr_window, self.hamming)
            fft = np.abs(np.fft.rfft(h))
            mel_vals = np.dot(self.filter_bank, np.transpose(fft))
            mel_vals = 20*np.log10(mel_vals)
            
            if(self.frame_num == 8):
                self.last_mel = mel_vals
            
            elif(self.frame_num > 8):
                rectified_diff = np.maximum(np.zeros((1,40)), (mel_vals - self.last_mel))
                next_onset = np.sum(rectified_diff)
                self.onset_env[self.index:self.index+len(self.gaussian)] += self.gaussian*next_onset
                self.last_mel = mel_vals
    

############################################################################################################

    #Calculates score for current time step and adds this value to the cumulative score array
    def calc_score(self):
        if(self.index < 750):
            self.cum_score[self.index] = self.onset_env[self.index]
        
        else:
            prev_beat = self.alpha*self.prev_beat_window + self.cum_score[self.index - 2*self.tempo_samples : self.index - self.tempo_samples/2]
            self.cum_score[self.index] = self.onset_env[self.index] + np.max(prev_beat)

  
        self.index += 1
   
############################################################################################################

    #Looks at most recent values of cumulative score to find whether we have encountered a peak in the cumulative score (i.e. a beat)
    #Criteria for a beat: 
    #   -cum_score must peak and then decrease for at least threshold number of time steps
    #   -The peak value in this local maximum must be greater than the largest previously encountered peak value
    def detect_peak(self):
        if(not self.peak_detected):
            if(self.cum_score[self.index - 1] > self.cum_score[self.index - 2]):
                self.num_increasing += 1
            
            elif(self.cum_score[self.index - 1] < self.cum_score[self.index - 2] and self.num_increasing > 0 and self.cum_score[self.index - 2] > self.prev_peak):
                self.num_increasing = 0
                self.peak_detected = True

            else:
                self.num_increasing = 0

        else:
            if(self.num_decreasing > self.threshold):
                self.num_beats += 1
                print("BEAT: " + (str)(self.num_beats) + "\n\n\n")
                self.peak_detected = False
                self.beats = np.append(self.beats, self.index)
                self.num_decreasing = 0
                self.prev_peak = self.cum_score[self.index - self.threshold - 3]

            elif(self.cum_score[self.index - 1] < self.cum_score[self.index - 2]):
                self.num_decreasing += 1


            else:
                self.num_decreasing = 0
                self.peak_detected = False

            
    
    
############################################################################################################

    #Performs autocorrelation on onset envelope to find the global tempo estimate and stores this estimate in tempo_samples
    def autocorrelate(self, max_lag):
        samples_per_beat = np.round(1.0 / (120.0/60.0/250.0)) #Defaults to 120 bpm or 125 samples per beat
        acr = np.zeros(max_lag)
        onset_lag = self.onset_env[0:self.index*2]

        for i in range(0,max_lag):
            onset_lag = np.append(np.zeros(i), self.onset_env[0:self.index*2 - i]) 
            acr[i] = np.correlate(self.onset_env[0:self.index*2], onset_lag)
        
        r = np.arange(1, max_lag + 1)
        window = np.exp(-0.5*np.power(np.log2(r/samples_per_beat), 2))
        win_acr = np.multiply(window, acr)
        z_pad = np.append(np.append([0],win_acr),[0])
        tps2 = (win_acr[0:(int)(np.ceil(len(win_acr)/2.0))] + 
            0.5*z_pad[1:len(win_acr) + 1:2] +
            0.25*z_pad[0:len(win_acr):2] +
            0.25*z_pad[2:len(win_acr) + 2:2])

        tps3 = (win_acr[0:(int)(np.ceil(len(win_acr)/3.0))] + 
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
            self.tempo_samples = startpd
        else:
            self.tempo_samples = startpd2

        print("Samples between beats: " + (str)(self.tempo_samples))
        print("Tempo (bpm): " + (str)(250.0 / self.tempo_samples * 60.0))

        r = np.arange(np.round(self.tempo_samples/2), self.tempo_samples*2, 1)
        self.prev_beat_window = -400*np.power((np.log(r/(float)(self.tempo_samples))), 2)
        self.prev_beat_window = np.fliplr([self.prev_beat_window])[0]



############################################################################################################

#Callback function used to perform all processing each time mic samples 4 ms of audio
def process(in_data, frame_count, time_info, status_flags):
    global tracker
    new_samples = np.fromstring(in_data, 'Int16')
    tracker.create_window(new_samples)
    
    if(tracker.index == 750):
        tracker.autocorrelate(tracker.index)
        tracker.alpha = np.std(tracker.onset_env[0:tracker.index])

    tracker.calc_onset()
    tracker.calc_score()

    if(tracker.index > 750):
        tracker.detect_peak()

    return(None, pyaudio.paContinue)

############################################################################################################

#Here's where execution begins
if __name__ == "__main__":
    global tracker
    p = pyaudio.PyAudio()
    tracker = BeatTracker()

    #Uncomment one of the two following lines depending on the version of python being used
    raw_input("Start music and press enter") #For Python 2.x
    #input("Start music and press enter") #For Python 3

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
    while(t2 - t1 < 30):
        time.sleep(0.1)
        t2 = time.time()
    stream.close()
    scipy.io.savemat('vars.mat', mdict = {'onset_env':tracker.onset_env, 'cum_score':tracker.cum_score, 'beats':tracker.beats})
        
 
