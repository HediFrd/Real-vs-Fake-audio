from opensoundscape import Audio, Spectrogram
from opensoundscape.spectrogram import MelSpectrogram
from pathlib import Path

# Load audio object
aud_obj = Audio.from_url("https://tinyurl.com/birds60s")
# aud_obj = Audio.from_file('/local/file/path')

# We can optionally resample, trim (in time), and bandpass (in freq) the audio object
# Not sure if we can resample at a higher sampling rate (i.e., interpolation?)
aud_obj = aud_obj#.resample(22000).trim(0, 5).bandpass(low_f=5500, high_f=9500, order=10)

# Create default spectrogram
spec_obj = Spectrogram.from_audio(aud_obj)
spec_obj.plot(inline=False)

# Spectrogram with more options:
# audio: Audio object
# window_type="hann": see scipy.signal.spectrogram docs
# window_samples: number of audio samples per spectrogram window (pixel)
#     - Defaults to 512 if window_samples and window_length_sec are None
#     - Note: cannot specify both window_samples and window_length_sec
# window_length_sec: length of a single window in seconds
#     - Note: cannot specify both window_samples and window_length_sec
#     - Warning: specifying this parameter often results in less efficient spectrogram computation because window_samples will not be a power of 2.
# overlap_samples: number of samples shared by consecutive windows
#     - Note: must not specify both overlap_samples and overlap_fraction
# overlap_fraction: fractional temporal overlap between consecutive windows
#     - Defaults to 0.5 if overlap_samples and overlap_fraction are None
#     - Note: cannot specify both overlap_samples and overlap_fraction
# fft_size: see scipy.signal.spectrogram's `nfft` parameter
# dB_scale: If True, rescales values to decibels, x=10*log10(x)
# scaling: ("spectrum" (V^2/Hz) or "density" (V^2)) see scipy.signal.spectrogram docs; Note that [default: "spectrum"] is different from scipy.signal.spectrogram's default
spec_obj = Spectrogram.from_audio(aud_obj, window_type="hann", window_samples=512, window_length_sec=None, overlap_samples=None,
                                  overlap_fraction=0.5, fft_size=None, dB_scale=True, scaling="spectrum")
spec_obj.plot(inline=False)

# Key note: I use "sp" for samples and remember Hz = samples per second
# Example: If the audio sample is 1s at a sample rate is 51200Hz, window_size=512, and overlap_fraction=0.0:
#     - Each window covers 512sp / 51200Hz = 0.01s of audio
#     - There are 1s / (0.01s * (1 - 0.0)) - ceil(1 - 0.0) = 100 total windows
#     - The lowest resolvable frequency from FFT of each window is 1 / 0.01s = 100Hz (with natural number multiples of this for each row of the spectrogram)
# Example: If the audio sample is 1s at a sample rate is 51200Hz, window_size=512, and overlap_fraction=0.5:
#     - Each window covers 512sp / 51200Hz = 0.01s of audio
#     - There are 1s / (0.01s * (1 - 0.5)) - ceil(1 - 0.5) = 199 total windows
#     - The lowest resolvable frequency from FFT of each window is 1 / 0.01s = 100Hz (with natural number multiples of this for each row of the spectrogram)
# Example: If the audio sample is 1s at a sample rate is 51200Hz, window_size=5120, and overlap_fraction=0.0:
#     - Each window covers 5120sp / 51200Hz = 0.1s of audio
#     - There are 1s / (0.1s * (1 - 0.0)) - ceil(1 - 0.0) = 10 total windows
#     - The lowest resolvable frequency from FFT of each window is 1 / 0.1s = 10Hz (with natural number multiples of this for each row of the spectrogram)

# We can also trim the spec object (in time), bandpass (in freq), and limit the dB range for plotting (default is (-100, 20), raising the min cuts out noise)
spec_obj.trim(1.7, 3.9).bandpass(min_f=5500, max_f=9500).plot(range=(-80, -20))
spec_obj.plot(inline=False)

# We can also use Mel Spectrograms which are scaled more for human perception
melspec_obj = MelSpectrogram.from_audio(aud_obj.trim(0, 20), window_samples=2048, n_mels=400)
melspec_obj.plot(inline=False)

# Inspecting spectrograms
spec_inspect = Spectrogram.from_audio(aud_obj)

# .times gives the times of the window centers, .frequencies gives the frequency values of each row
print(f"The first few times: {spec_inspect.times[0:5]}")
print(f"The first few frequencies: {spec_inspect.frequencies[0:5]}")

# The whole spectrogram can be obtained as a numpy array
print(spec_inspect.spectrogram)

# Summing across each column gives the amplitude as a function of time
spec_inspect.amplitude()

# Saving the spectrogram as an image with specified shape - can invert the color (white is more intense)
spec_img = spec_obj.to_image(shape=(224, 224), invert=False)
spec_img.save(Path("./saved_spectrogram.png"))

