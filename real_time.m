%%Read in audio file
filename = 'syncopated_groove.wav';
[s, Fs] = audioread(filename);
s = s(:,1);
g = gcd(8000,Fs);
s = resample(s,8000/g,Fs/g);
Fs = 8000;
onset_env = [];
cum_score = [];
D = [];
alpha = 100;
tempo = 165; %bpm
tempo_samples = round(250/(tempo/60)); %tempo in units of samples
prev_beat_weighting_window = (-(log10((round(tempo_samples/2):(tempo_samples*2))/tempo_samples)).^2)';

%%Take first 32 ms and compute MFC
w = s(1:256);
%[stft, freq, t] = spectrogram(w, length(w), 0, length(w), Fs);
stft = fft(w.*hamming(length(w)));
stft = abs(stft(1:length(stft)/2+1));
mel_filters = fft2melmx(length(stft), Fs, 40);
mel_filters = mel_filters(:,1:length(stft));
mel_vals = mel_filters * stft;
D = [D mel_vals];
mel_vals = 20*log10(max(1e-10, mel_vals));

%%While there is more audio, hop forward 4 ms and and compute the new
%%window's MFC.  Compute first difference over time of the MFCs and store value
%%in onset_env
for a = 256:32:length(s)
    a
    w = [w(33:end); s(a-31:a)];
    
    if length(w) ~= 256
        break;
    end
    
    %[stft, freq, t] = spectrogram(w, length(w), 0, length(w), Fs);
    stft = fft(w.*hamming(length(w)));
    stft = abs(stft(1:length(stft)/2+1));
    
    next_mel_vals = mel_filters * stft;
    D = [D next_mel_vals];
    next_mel_vals = 20*log10(max(1e-10,next_mel_vals));
    d = next_mel_vals - mel_vals;
    onset_env = [onset_env; mean(max(0,d))];
    mel_vals = next_mel_vals;
    
    %%Now that the next value in the onset envelope is calculated, we use
    %%that value and search previous values to get the next cumulative
    %%score value
    if length(onset_env) <= 2*tempo_samples+1
        cum_score = [cum_score; onset_env(end)];
    else
        prev_beat_range = cum_score(end - tempo_samples*2:end - round(tempo_samples/2));
        new_score = onset_env(end) + max(prev_beat_range + alpha*prev_beat_weighting_window);
        cum_score = [cum_score; new_score];
    end
end

plot(cum_score);
hold on
b = importdata('./syncopated_groove.txt');
b = 250*b;
for a = b
    plot([a,a],[min(cum_score), max(cum_score)],'g-');
end
hold off
% plot(onset_env);
% hold on;
% b_t = b*(1/0.004);
% for a = b_t
%     plot([a a], [min(onset_env) max(onset_env)], 'r-');
% end
    




