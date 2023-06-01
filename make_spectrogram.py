import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# Set the sample rate and duration
sample_rate = 44100
duration = 4

# Record audio from the microphone
print("Recording started...")
audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
sd.wait()
print("Recording stopped.")

# Convert audio data to spectrogram
spectrogram, frequencies, times, _ = plt.specgram(audio_data[:, 0], Fs=sample_rate)

# Plot the spectrogram
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')

# Save the spectrogram as an image
plt.savefig('C:/Users/Bruker/Documents/wael/train/3_0.png')

# Show the spectrogram
plt.show()
