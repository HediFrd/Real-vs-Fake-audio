import matplotlib.pyplot as plt
import numpy as np
from opensoundscape import Audio, Spectrogram
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tvt

def gather_audio_metadata(aud_dir, n_samples=None):

    duration = []
    frames = []
    sample_rate = []

    aud_files = os.listdir(aud_dir)
    if n_samples is not None:
        random.shuffle(aud_files)
        aud_files = aud_files[:n_samples]

    # Loop through each audio file, load it, and extract metadata
    for aud_file in aud_files:
        if os.path.splitext(aud_file)[1] not in [".wav", ".mp3"]:
            continue

        aud_obj = Audio.from_file(os.path.join(aud_dir, aud_file))
        duration.append(aud_obj.metadata["duration"])
        frames.append(aud_obj.metadata["frames"])
        sample_rate.append(aud_obj.metadata["samplerate"])

    return duration, frames, sample_rate


def process_audio(aud_obj, resample_rate=None, loop_to=None, t_min=None, t_max=None, f_min=None, f_max=None):

    old_meta = {"duration": aud_obj.duration, "frames": len(aud_obj.samples), "samplerate": aud_obj.sample_rate}

    # Resample audio to specified sample_rate
    if resample_rate is not None:
        aud_obj = aud_obj.resample(resample_rate)

    # Loop the audio until it achieves the desired duration
    if loop_to is not None and aud_obj.duration < loop_to:
        aud_obj = aud_obj.loop(length=loop_to)

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

def process_spectrogram(spec_obj, t_min=None, t_max=None, f_min=None, f_max=None, dB_clip_min=None, dB_clip_max=None, dB_rescale_min=None, dB_rescale_max=None):

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
    if dB_clip_min is not None or dB_clip_max is not None:
        dB_clip_min = -100 if dB_clip_min is None else dB_clip_min
        dB_clip_max = -20 if dB_clip_max is None else dB_clip_max
        spec_obj = spec_obj.limit_range(min=dB_clip_min, max=dB_clip_max)

    # Rescale image values to
    if dB_rescale_min is not None or dB_rescale_max is not None:
        dB_rescale_min = -100 if dB_rescale_min is None else dB_rescale_min
        dB_rescale_max = -20 if dB_rescale_max is None else dB_rescale_max
        inp_min = np.min(spec_obj.spectrogram)
        inp_max = np.max(spec_obj.spectrogram)
        spec_obj = spec_obj.linear_scale(feature_range=(dB_rescale_min, dB_rescale_max), input_range=(inp_min, inp_max))

    return spec_obj

def predef_spec_size(aud_obj, max_freq, n_freqs, n_windows, target_duration=None, overlap_correction=0.9):
    # max_freq [Hz] Max frequency plotted spectrogram
    # n_freqs [-] Number of frequencies utilized by spectrogram (y-axis length)
    # n_windows [-] Number of windows utilized by spectrogram (x-axis length)
    # target_duration [s] Target length of audio - if None, uses original audio sample duration
    # overlap_correction [samples] Sample shift, used to preempt the multiple floor operations used in calculating n_windows in spectrogram

    # Calculate per-window statistics
    sample_rate = max_freq * 2.  # [Hz] Necessary sampling rate to achieve desired max frequency
    freq_interval = max_freq / (n_freqs - 1)  # [Hz] Frequency interval of spectrogram image
    window_size = 2 * (n_freqs - 1)  # [samples] Number of samples spanned by each window
    window_time = window_size / sample_rate  # [s] Length of time spanned by each window

    # Calculate min and max durations permissible
    max_overlap = window_size - 1  # [samples] Maximum possible number of overlap samples
    min_samples = window_size + n_windows  # [samples] Minimum number of samples required
    min_duration = min_samples / sample_rate  # [s] Minimum audio duration
    min_overlap = 0
    max_samples = window_size * n_windows  # [samples] Maximum samples permissible
    max_duration = max_samples / sample_rate  # [s] Maximum audio duration

    # Corrections to audio sample rate and duration
    resample_rate = sample_rate if aud_obj.sample_rate != sample_rate else None
    if target_duration is None:
        target_duration = aud_obj.duration
    target_duration = max(target_duration, min_duration)
    target_duration = min(target_duration, max_duration)
    target_duration = int(target_duration * sample_rate) / sample_rate  # Floors the target duration to the nearest divisible unit

    aud_obj, _, _ = process_audio(aud_obj, resample_rate=resample_rate, loop_to=target_duration, t_max=target_duration)

    # Set the number of overlap samples to meet the desired n_windows and duration
    target_samples = int(target_duration * sample_rate)  # [samples] Target number of samples
    overlap_samples = window_size - (target_samples - window_size) / (n_windows - 1)  # Number of overlap samples required to meet given constraints
    overlap_samples_final = int(overlap_samples + overlap_correction)  # Final overlap samples considering numerical issues

    spec_obj = Spectrogram.from_audio(aud_obj, window_samples=window_size, overlap_samples=overlap_samples_final)

    # It is impossible to guarantee that the correct number of windows is always achieved due to other constraints (like n_freqs and duration)
    # Minorly adjust width of spectrogram to meet the desired n_windows
    n_windows_gap = n_windows - spec_obj.spectrogram.shape[1]
    if abs(n_windows_gap) > 5:
        print(f"Warning: Spectrogram width disagrees with desired by {n_windows_gap} windows. Recommend checking duration and n_windows settings.")
    if n_windows_gap > 0:
        # Pad minimum values at end to meet n_windows requirement
        new_times = np.cumsum(np.ones(n_windows_gap)) * (spec_obj.times[1] - spec_obj.times[0]) + spec_obj.times[-1]
        times_arr = np.concat([spec_obj.times, new_times], axis=0)
        new_spec = np.ones((n_freqs, n_windows_gap)) * np.min(spec_obj.spectrogram)
        spec_arr = np.concat([spec_obj.spectrogram, new_spec], axis=1)
        spec_obj = spec_obj._spawn(times=times_arr, spectrogram=spec_arr)
    elif n_windows_gap < 0:
        # Sever extra columns
        times_arr = spec_obj.times[:n_windows]
        spec_arr = spec_obj.spectrogram[:, :n_windows]
        spec_obj = spec_obj._spawn(times=times_arr, spectrogram=spec_arr)

    return spec_obj

def rgb_spec_pipeline(aud_obj, max_freq_list, n_freqs, n_windows, target_duration, overlap_correction, dB_min_list, dB_max_list):
    # max_freq_list is max_freq for each RGB
    # dB_min_list is dB_min for each RGB
    # dB_max_list is dB_max for each RGB
    # Other variables are explained in predef_spec_size
    # If max_freq_list, dB_min_list, and dB_max_list are length 1 or all of the same values, the final image will be grayscale

    assert len(max_freq_list) == len(dB_min_list)
    assert len(max_freq_list) == len(dB_max_list)

    # Loop through each item for RGB and then get the corresponding spectrogram image
    spec_img_list = []
    for ii in range(len(max_freq_list)):
        spec_obj = predef_spec_size(aud_obj, max_freq=max_freq_list[ii], n_freqs=n_freqs, n_windows=n_windows,
                                    target_duration=target_duration, overlap_correction=overlap_correction)
        if dB_min_list[ii] is not None and dB_max_list[ii] is not None:
            spec_obj = process_spectrogram(spec_obj, dB_clip_min=dB_min_list[ii], dB_clip_max=dB_max_list[ii],
                                           dB_rescale_min=dB_min_list[ii], dB_rescale_max=dB_max_list[ii])
            spec_img = spec_obj.to_image(shape=(n_freqs, n_windows), range=(dB_min_list[ii], dB_max_list[ii]), invert=False)
        else:
            spec_img = spec_obj.to_image(shape=(n_freqs, n_windows), invert=False)
        spec_img_list.append(spec_img)

    # Concatenate spectrogram images to form final RGB image
    final_arr = np.stack([np.asarray(spec_img) for spec_img in spec_img_list], axis=2)
    final_img = Image.fromarray(final_arr.squeeze())

    return final_img


class Spectrogram_Dataset(Dataset):
    """Dataset for loading and processing audio files then converting to spectrograms"""

    def __init__(self, real_root, fake_root, aud_dur_range=[5, 5], aud_rand_start=True, spec_size=(256, 256), max_freq_list=[8000], dB_min_list=[None], dB_max_list=[None], transform=None):
        real_files = os.listdir(real_root)
        self.real_paths = sorted([os.path.join(real_root, real_file) for real_file in real_files])
        self.n_real = len(self.real_paths)
        if fake_root is not None:
            fake_files = os.listdir(fake_root)
            self.fake_paths = sorted([os.path.join(fake_root, fake_file) for fake_file in fake_files])
            self.n_fake = len(self.fake_paths)
        else:
            self.n_fake = 0
        self.aud_dur_range = aud_dur_range
        self.aud_rand_start = aud_rand_start
        self.spec_size = spec_size
        self.max_freq_list = max_freq_list
        self.dB_min_list = dB_min_list
        self.dB_max_list = dB_max_list

        if transform is None:
            self.transform = tvt.Compose([tvt.ToTensor()])
        else:
            self.transform = transform

    def __len__(self):
        return self.n_real + self.n_fake

    def __getitem__(self, idx):

        # Get file and label
        if idx < self.n_real:
            aud_path = self.real_paths[idx]
            label = torch.tensor(1).long()
        else:
            aud_path = self.fake_paths[idx - self.n_real]
            label = torch.tensor(0).long()

        # Process audio to specified duration
        try:
            aud_obj = Audio.from_file(aud_path)
        except:
            # Return zeros tensor with fake label
            return torch.zeros((3, self.spec_size[0], self.spec_size[1])), torch.tensor(0).long()

        aud_tgt_dur = torch.rand(size=(1,)).item() * (self.aud_dur_range[1] - self.aud_dur_range[0]) + self.aud_dur_range[0]
        if aud_obj.duration <= aud_tgt_dur:
            # Just loop to meet target duration
            aud_obj, _, _ = process_audio(aud_obj, loop_to=aud_tgt_dur)
        elif self.aud_rand_start:
            # Sample random length inside audio of target duration
            dur_frames = int(aud_obj.sample_rate * aud_tgt_dur)
            start_frame = torch.randint(low=0, high=(len(aud_obj.samples) - dur_frames), size=(1,))
            aud_obj, _, _ = process_audio(aud_obj, t_min=start_frame / aud_obj.sample_rate,
                                          t_max=(start_frame + dur_frames) / aud_obj.sample_rate)
        else:
            # Get clip at beginning of audio
            aud_obj, _, _ = process_audio(aud_obj, t_max=aud_tgt_dur)

        # Make the spectrogram image
        # Note that the spec pipeline will possibly redo the aud_obj processing for target duration
        # This is necessary to get the spectrogram of the correct size and scale
        # Essentially need to set aud_tgt_dur and aud_samp_rate intelligently to ensure they don't need to be recast unexpectedly
        spec_img = rgb_spec_pipeline(aud_obj, max_freq_list=self.max_freq_list, n_freqs=self.spec_size[0], n_windows=self.spec_size[1],
                                     target_duration=aud_tgt_dur, overlap_correction=0.9, dB_min_list=self.dB_min_list, dB_max_list=self.dB_max_list)
        spec_img = self.transform(spec_img.convert("RGB"))

        return spec_img, label

class Spectrogram_Dataset_Custom(Dataset):
    """Dataset for loading and processing audio files then converting to spectrograms"""

    def __init__(self, real_root, fake_root, aud_dur_range=[5, 5], aud_rand_start=True, spec_size=(256, 256), max_freq_list=[8000], dB_min_list=[None], dB_max_list=[None], transform=None):
        real_files = os.listdir(real_root)
        self.real_paths = sorted([os.path.join(real_root, real_file) for real_file in real_files])
        self.n_real = len(self.real_paths)
        if fake_root is not None:
            fake_files = os.listdir(fake_root)
            self.fake_paths = sorted([os.path.join(fake_root, fake_file) for fake_file in fake_files])
            self.n_fake = len(self.fake_paths)
        else:
            self.n_fake = 0
        self.aud_dur_range = aud_dur_range
        self.aud_rand_start = aud_rand_start
        self.spec_size = spec_size
        self.max_freq_list = max_freq_list
        self.dB_min_list = dB_min_list
        self.dB_max_list = dB_max_list

        if transform is None:
            self.transform = tvt.Compose([tvt.ToTensor()])
        else:
            self.transform = transform

    def __len__(self):
        return self.n_real + self.n_fake

    def __getitem__(self, idx):

        # Get file and label
        if idx < self.n_real:
            aud_path = self.real_paths[idx]
            label = torch.tensor(1).long()
        else:
            aud_path = self.fake_paths[idx - self.n_real]
            label = torch.tensor(0).long()

        # Process audio to specified duration
        try:
            aud_obj = Audio.from_file(aud_path)
        except:
            # Return zeros tensor with fake label
            return torch.zeros((3, self.spec_size[0], self.spec_size[1])), torch.tensor(0).long()

        #################################################
        out_folder = r"D:\Deep_Detect_Spec\training\fake"
        #################################################

        aud_tgt_dur = torch.rand(size=(1,)).item() * (self.aud_dur_range[1] - self.aud_dur_range[0]) + self.aud_dur_range[0]
        if aud_obj.duration <= aud_tgt_dur:
            # Just loop to meet target duration
            aud_obj, _, _ = process_audio(aud_obj, loop_to=aud_tgt_dur)

            # Make the spectrogram image
            # Note that the spec pipeline will possibly redo the aud_obj processing for target duration
            # This is necessary to get the spectrogram of the correct size and scale
            # Essentially need to set aud_tgt_dur and aud_samp_rate intelligently to ensure they don't need to be recast unexpectedly
            spec_img = rgb_spec_pipeline(aud_obj, max_freq_list=self.max_freq_list, n_freqs=self.spec_size[0], n_windows=self.spec_size[1],
                                         target_duration=aud_tgt_dur, overlap_correction=0.9, dB_min_list=self.dB_min_list, dB_max_list=self.dB_max_list)
            spec_img.convert("RGB").save(os.path.join(out_folder, os.path.splitext(os.path.basename(aud_path))[0] + ".png"))

        elif self.aud_rand_start:
            for kk in range(3):

                # Sample random length inside audio of target duration
                dur_frames = int(aud_obj.sample_rate * aud_tgt_dur)
                start_frame = torch.randint(low=0, high=(len(aud_obj.samples) - dur_frames), size=(1,))
                # aud_obj, _, _ = process_audio(aud_obj, t_min=start_frame / aud_obj.sample_rate,
                #                               t_max=(start_frame + dur_frames) / aud_obj.sample_rate)
                aud_obj_new, _, _ = process_audio(aud_obj, t_min=start_frame / aud_obj.sample_rate,
                                              t_max=(start_frame + dur_frames) / aud_obj.sample_rate)

                # Make the spectrogram image
                # Note that the spec pipeline will possibly redo the aud_obj processing for target duration
                # This is necessary to get the spectrogram of the correct size and scale
                # Essentially need to set aud_tgt_dur and aud_samp_rate intelligently to ensure they don't need to be recast unexpectedly
                spec_img = rgb_spec_pipeline(aud_obj_new, max_freq_list=self.max_freq_list, n_freqs=self.spec_size[0], n_windows=self.spec_size[1],
                                             target_duration=aud_tgt_dur, overlap_correction=0.9, dB_min_list=self.dB_min_list, dB_max_list=self.dB_max_list)
                spec_img.convert("RGB").save(os.path.join(out_folder, os.path.splitext(os.path.basename(aud_path))[0] + f"_{kk}.png"))

        else:
            # Get clip at beginning of audio
            aud_obj, _, _ = process_audio(aud_obj, t_max=aud_tgt_dur)

        # # Make the spectrogram image
        # # Note that the spec pipeline will possibly redo the aud_obj processing for target duration
        # # This is necessary to get the spectrogram of the correct size and scale
        # # Essentially need to set aud_tgt_dur and aud_samp_rate intelligently to ensure they don't need to be recast unexpectedly
        # spec_img = rgb_spec_pipeline(aud_obj, max_freq_list=self.max_freq_list, n_freqs=self.spec_size[0], n_windows=self.spec_size[1],
        #                              target_duration=aud_tgt_dur, overlap_correction=0.9, dB_min_list=self.dB_min_list, dB_max_list=self.dB_max_list)
        # spec_img = self.transform(spec_img.convert("RGB"))
        #
        # return spec_img, label
        return torch.tensor([0.]), torch.tensor([0.])

def file_quarantine(aud_dir):
    quarantine_list = []
    aud_files = os.listdir(aud_dir)
    for aud_file in aud_files:
        aud_path = os.path.join(aud_dir, aud_file)
        try:
            Audio.from_file(aud_path)
        except:
            print(f"Error loading {aud_path}")
            quarantine_list.append(aud_path)

    return quarantine_list

if __name__ == "__main__":
    print("Warning: Running __main__ from misc_utils.py")

    # aud_dir = r"D:\Deep_Detect\dataset\training\real"
    # _ = file_quarantine(aud_dir)

    # Using the gather_audio_metadata function to gather metadata for an entire directory
    # aud_dir = r"D:\Deep_Detect\dataset\training\real"
    # dur, fr, sr = gather_audio_metadata(aud_dir, n_samples=2000)
    # plt.hist(dur, bins=20)
    # plt.show()
    # print(np.mean(dur), np.std(dur), np.max(dur), np.min(dur))
    # print(np.mean(sr), np.std(sr), np.max(sr), np.min(sr))

    # User inputs
    # aud_path = r"D:\FoRA\for-original\for-original\testing\real\file2.wav"
    # aud_obj = Audio.from_file(aud_path)
    # out_path = r"outputs/saved_spec5.png"
    # n_freqs = 256
    # n_windows = 256
    # target_duration = 5
    # overlap_correction = 0.9
    # max_freq_list = [8000, 8000, 2000]
    # dB_min_list = [None, -100, None]
    # dB_max_list = [None, -60, None]

    # Simple examples of calling each function
    # aud_obj = Audio.from_file(aud_path)
    # aud_obj, old_meta, new_meta = process_audio(aud_obj, loop_to=2, t_max=2)
    # spec_obj = Spectrogram.from_audio(aud_obj, window_type="hann", window_samples=510, overlap_samples=386, dB_scale=True, scaling="spectrum")
    # dB_min, dB_max = -100, -40
    # spec_obj = process_spectrogram(spec_obj, dB_clip_min=dB_min, dB_clip_max=dB_max)
    # # spec_obj.times
    # # spec_obj.frequencies
    # # spec_obj.amplitude()
    # # Note the image range should correspond to the dB range in process_spectrogram or else you get brightness shifts
    # spec_img = spec_obj.to_image(shape=(256, 256), range=(dB_min, dB_max), invert=False).save(out_path)

    # aud_obj = Audio.from_file(aud_path)

    # Example of generating a spectrogram of a predefined size and scale
    # spec_obj = predef_spec_size(aud_obj, max_freq=8000, n_freqs=n_freqs, n_windows=n_windows, target_duration=target_duration, overlap_correction=overlap_correction)
    # spec_img = spec_obj1.to_image(shape=(n_freqs, n_windows), invert=False).save(out_path)

    # Example of using the pipeline to generate an RGB image of multiple spectrograms of identical size
    # rgb_spec_img = rgb_spec_pipeline(aud_obj, max_freq_list, n_freqs, n_windows, target_duration, overlap_correction, dB_min_list, dB_max_list)
    # rgb_spec_img.save(out_path)

    # Dataset settings
    # train_real_root = r"D:\Deep_Detect\dataset\training\fake"
    # train_fake_root = None  # r"D:\Deep_Detect\dataset\training\fake"
    # train_dur_range = [5, 5]  # Range of audio target durations for training
    # test_dur_range = [5, 5]  # Range of audio target durations for testing
    # aud_rand_start = True  # Whether to start the audio sampling at a random point
    # spec_size = (224, 224)  # Number of frequencies and number of windows in the spectrogram
    # max_freq_list = [8000, 4000, 2000]  # Maximum frequency on the spectrogram scale - list of 1 or 3 elements
    # dB_min_list = [None, None, None]  # Decibel minimum - list of 1 or 3 elements, None means no limit applied
    # dB_max_list = [None, None, None]  # Decibel maximum - list of 1 or 3 elements, None means no limit applied
    #
    # # Training settings
    # batch_size = 1  # Training and evaluation batch size
    #
    # from torch.utils.data import DataLoader
    # train_dataset = Spectrogram_Dataset_Custom(train_real_root, train_fake_root, train_dur_range, aud_rand_start, spec_size, max_freq_list, dB_min_list, dB_max_list)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #
    # out_folder = r"D:\Deep_Detect_Spec\training\real"
    # for ii, (images, labels) in enumerate(train_dataloader):
    #     continue
