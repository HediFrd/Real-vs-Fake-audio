import numpy as np
from opensoundscape import Audio, Spectrogram
import os

def gather_audio_metadata(aud_dir):

    duration = []
    frames = []
    sample_rate = []

    # Loop through each audio file, load it, and extract metadata
    for aud_file in os.listdir(aud_dir):
        if os.path.splitext(aud_file)[1] not in [".wav"]:
            continue

        aud_obj = Audio.from_file(os.path.join(aud_dir, aud_file))
        duration.append(aud_obj.metadata["duration"])
        frames.append(aud_obj.metadata["frames"])
        sample_rate.append(aud_obj.metadata["samplerate"])

    return duration, frames, sample_rate


def process_audio(aud_obj, resample_rate=None, min_dur=None, t_min=None, t_max=None, f_min=None, f_max=None):

    old_meta = {"duration": aud_obj.duration, "frames": len(aud_obj.samples), "samplerate": aud_obj.sample_rate}

    # Resample audio to specified sample_rate
    if resample_rate is not None:
        aud_obj = aud_obj.resample(resample_rate)

    # Loop the audio until it achieves the desired duration
    if min_dur is not None and aud_obj.duration < min_dur:
        aud_obj = aud_obj.loop(length=min_dur)

    # Trim audio length (time) to specified limits
    if t_min is not None or t_max is not None:
        t_min = 0 if t_min is None else t_min
        t_max = 1e6 if t_max is None else t_max
        aud_obj = aud_obj.trim(start_time=t_min, end_time=t_max)

    # If f_max is not None but above sample_rate / 2, we don't want to mess with that bound if possible, so reset it
    if f_max is not None and f_max >= aud_obj.sample_rate / 2:
        f_max = None

    # Bandpass filter the audio
    if f_min is not None or f_max is not None:
        f_min = 1 if f_min is None else f_min
        if f_max is None or f_max >= aud_obj.sample_rate / 2:
            f_max = aud_obj.sample_rate / 2 - 1
        aud_obj = aud_obj.bandpass(low_f=f_min, high_f=f_max, order=10)

    new_meta = {"duration": aud_obj.duration, "frames": len(aud_obj.samples), "samplerate": aud_obj.sample_rate}

    return aud_obj, old_meta, new_meta

def process_spectrogram(spec_obj, t_min=None, t_max=None, f_min=None, f_max=None, dB_min=None, dB_max=None):

    # Trim spectrogram length (time) to specified limits
    if t_min is not None or t_max is not None:
        t_min = 0 if t_min is None else t_min
        t_max = 1e6 if t_max is None else t_max
        spec_obj = spec_obj.trim(start_time=t_min, end_time=t_max)

    # Bandpass filter the spectrogram
    # Note that the spectrogram bandpass function handles out-of-bounds frequencies automatically
    if f_min is not None or f_max is not None:
        f_min = 0 if f_min is None else f_min
        f_max = 1e6 if f_max is None else f_max
        spec_obj = spec_obj.bandpass(min_f=f_min, max_f=f_max)

    # Clip the intensity to given range
    # Defaults are [-100, -20] for this and for saving/plotting, so I maintain those
    if dB_min is not None or dB_max is not None:
        dB_min = -100 if dB_min is None else dB_min
        dB_max = -20 if dB_max is None else dB_max
        spec_obj = spec_obj.limit_range(min=dB_min, max=dB_max)

    return spec_obj


if __name__ == "__main__":
    print("Warning: Running __main__ from misc_utils.py")

    # aud_path = r"D:\FoRA\for-original\for-original\testing\real\file2.wav"
    # out_path = r"outputs/saved_spec4.png"
    # dB_min = -80
    # dB_max = None
    # dB_min = -100 if dB_min is None else dB_min
    # dB_max = -20 if dB_max is None else dB_max
    #
    # aud_obj = Audio.from_file(aud_path)
    # aud_obj, old_meta, new_meta = process_audio(aud_obj)#, min_dur=3, t_min=1., f_min=100, f_max=9000)
    # spec_obj = Spectrogram.from_audio(aud_obj, window_type="hann", window_samples=512, overlap_fraction=None, dB_scale=True, scaling="spectrum")
    # spec_obj = process_spectrogram(spec_obj, dB_min=dB_min)
    # # spec_obj.times
    # # spec_obj.frequencies
    # # spec_obj.amplitude()
    # # Note the image range should correspond to the dB range in process_spectrogram or else you get brightness shifts
    # output_img = spec_img = spec_obj.to_image(shape=(256, 256), range=(dB_min, dB_max), invert=False).save(out_path)

    # aud_dir = r"D:\FoRA\for-original\for-original\testing\real"
    # dur, fr, sr = gather_audio_metadata(aud_dir)
