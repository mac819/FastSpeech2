import os
import yaml
from pathlib import Path

import pywt
import torch
import torchaudio
import numpy as np
import pyworld as pw
from tqdm import tqdm
import scipy.io.wavfile as wav
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy.interpolate import interp1d


PACKAGE_PATH = Path(__file__).parent.parent
DATA_PATH = PACKAGE_PATH.parent / 'data'
AUDIO_TEXT_DATA = DATA_PATH / 'interim' / 'in_corpus'



# Alignment calculation from MFA TextGrids
# Alignment
# def get_alignment(ph_intervals, sample_rate, hop_length, silence_token=_silences[-1]):
#     phoneme_lst = []
#     duration_lst = []
#     prev_idx = 0
#     curr_idx = 1
#     audio_start_time = 0
#     audio_end_time = 0
#     while curr_idx < len(ph_intervals):
#         curr_interval = ph_intervals[curr_idx]
#         prev_interval = ph_intervals[prev_idx]
#         curr_start_time, curr_end_time, curr_phon = curr_interval.start_time, curr_interval.end_time, curr_interval.text
#         prev_start_time, prev_end_time, prev_phon = prev_interval.start_time, prev_interval.end_time, prev_interval.text
    
#         if (prev_idx == 0) and prev_phon != silence_token:
#             phoneme_lst.append(prev_phon)
#             duration = int(
#                 np.round(prev_end_time * sample_rate / hop_length) - 
#                 np.round(prev_start_time * sample_rate / hop_length)
#             )
#             duration_lst.append(duration)

#             # Setting audio start and end time
#             audio_start_time = prev_start_time
#             audio_end_time = prev_end_time
            
    
#         # Trimming trailing silence
#         if (len(phoneme_lst) == 0) and curr_phon == silence_token:
#             prev_idx += 1
#             curr_idx += 1

#             # Setting audio start and end time
#             audio_start_time = curr_end_time
#             audio_end_time = curr_end_time
#             continue
    
#         # Silence token for non-voiced frames    
#         if curr_phon in phoneme_vocab:
#             if curr_start_time != prev_end_time:
#                 # Add silence token in between the previous and current tokens
#                 phoneme_lst.append(silence_token)
#                 duration = int(
#                     np.round(curr_start_time * sample_rate / hop_length) - 
#                     np.round(prev_end_time * sample_rate / hop_length)
#                 )
#                 duration_lst.append(duration)

#             phoneme_lst.append(curr_phon)
#             duration = int(
#                 np.round(curr_end_time * sample_rate / hop_length) - 
#                 np.round(curr_start_time * sample_rate / hop_length)
#             )
#             duration_lst.append(duration)

#             audio_end_time = curr_end_time
        
    
#         prev_idx += 1
#         curr_idx += 1

#     assert len(phoneme_lst) == len(duration_lst), "Length of phoneme and duration list should be same"

#     return phoneme_lst, duration_lst, audio_start_time, audio_end_time


# Calculate Mel-Spectrogram
def calc_mel_spectrogram(filepath, config):
    
    n_fft = config['n_fft'] # 1024
    win_length = None # Same as n_fft
    hop_length = config['hop_length'] # 512
    n_mels = config['n_mels'] # 128
    sample_rate = config['sample_rate']

    # Load audio
    audio_arr, sr = torchaudio.load(filepath)

    # Resample audio if audio sample_rate doesn't match with target sample_rate
    if sr != sample_rate:
        audio_arr = F.resample(
            waveform=audio_arr, 
            orig_freq=sr,
            new_freq=sample_rate
        )
    
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )
    
    melspec = mel_spectrogram(audio_arr)
    return melspec# , audio_arr, sample_rate


# Calculate Pitch Spectrogram
def calc_pitch_spectrogram(filepath, config: dict):
    hop_length = config['hop_length']
    sample_rate = config['sample_rate']
    # CWT params
    tau = config['tau']
    band = config['band']
    
    fs, x = wav.read(filepath)
    x = x.astype(np.float64)

    # Extract raw F0 using WORLD
    _f0, t = pw.harvest(x, fs, frame_period=hop_length / sample_rate * 1000)
    f0 = pw.stonemask(x, _f0, t, fs)

    # Interpolate unvoiced frames
    valid = f0 > 0
    f0_interp = interp1d(t[valid], f0[valid], kind='linear', fill_value='extrapolate')
    f0_filled = f0_interp(t)

    # Step 3: Normalize pitch contour (log scale)
    f0_filled[f0_filled <= 0] = 1e-6 # Giving small value to the negative interpolated values
    f0_log = np.log(f0_filled)
    f0_mean, f0_std = np.mean(f0_log), np.std(f0_log)
    f0_norm = (f0_log - f0_mean) / f0_std

    # Apply Continuous Wavelet Transform

    scale = [2**(i+1)*tau for i in range(band)]
    sample_rate = 1 / tau
    scale_factor = [int(np.round(s * sample_rate)) for s in scale]
    # print(scale_factor)
    
    coeffs, freqs = pywt.cwt(f0_norm, scale_factor, 'mexh')
    return coeffs, f0_norm, f0_mean, f0_std, t


# Calculate Energy
def calculate_stft_energy(filepath, config):
    """
    Calculate energy using L2-norm of STFT as described in FastSpeech2 paper
    
    Args:
    - waveform: Input audio tensor [1, num_samples] or [num_samples]
    - sample_rate: Sampling rate of the audio
    - n_fft: Number of FFT components
    - hop_length: Number of samples between successive frames
    - win_length: Each frame of audio will be windowed with a window of this length
    
    Returns:
    - energy: Tensor of energy values for each frame
    """

    sample_rate = config['sample_rate']
    hop_length = config['hop_length']
    n_fft = config['n_fft']
    win_length = config['win_length']

    waveform, sr = torchaudio.load(filepath)

    if sr != sample_rate:
        waveform = F.resample(
            waveform=waveform, 
            orig_freq=sr,
            new_freq=sample_rate
        )

    # Ensure waveform is 1D
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)
    
    # Create window
    window = torch.hann_window(win_length)
    
    # Compute STFT
    # Use torch.stft directly from torch.signal
    stft_complex = torch.stft(
        waveform, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        return_complex=False  # Returns real and imaginary parts separately
    )
    
    # Calculate magnitude (L2-norm)
    # For non-complex output, manually calculate magnitude
    # stft_complex shape: [num_freq, num_frames, 2] (real and imaginary parts)
    stft_magnitude = torch.sqrt(stft_complex[..., 0]**2 + stft_complex[..., 1]**2)
    
    # Sum across frequency bins to get energy per frame
    frame_energy = torch.sum(stft_magnitude**2, dim=0)
    
    # Optional: Normalize energy
    frame_energy = frame_energy / (frame_energy.max() + 1e-8)
    
    return frame_energy


if __name__=="__main__":
    print(f"Package path: {PACKAGE_PATH}")
    feature_dir = DATA_PATH / 'processed' / 'lj_speech_feature'

    # Create feature directory
    mel_feature_dir = feature_dir / 'mel_spec'
    os.makedirs(str(mel_feature_dir), exist_ok=True)
    
    pitch_feature_dir = feature_dir / 'pitch'
    os.makedirs(str(pitch_feature_dir), exist_ok=True)
    
    energy_feature_dir = feature_dir / 'energy'
    os.makedirs(str(energy_feature_dir), exist_ok=True)
    
    # Reading Config
    with open(PACKAGE_PATH / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(config)
    lj_data = DATA_PATH / 'raw' / 'LJSpeech-1.1'
    lj_medata_path = lj_data / 'metadata.csv'
    with open(lj_medata_path, 'r') as f:
        for lines in tqdm(f.readlines()):
            # print(lines.split("|"))
            fname, text, norm_text = lines.split("|")
            # if fname in ['LJ006-0011', 'LJ007-0005', 'LJ011-0177', 'LJ016-0338', 'LJ028-0259']:
            audio_path = lj_data / 'wavs' / f'{fname}.wav'
            if audio_path.exists():
                # print(f"Processing features for file: {fname}")
                audio_path = str(audio_path)
                # Mel-spectrogram
                mel_config = config['preprocessing']['mel_spectrogram']
                # print(f"mel-config: {mel_config}")
                mel_spec = calc_mel_spectrogram(filepath=audio_path, config=mel_config)
                mel_outpath = mel_feature_dir / f'mel_{fname}.npy'
                _ = np.save(mel_outpath, mel_spec)
                # print(f"Mel-spec shape: {mel_spec.shape}")
                # Pitch Feauters
                pitch_config = config['preprocessing']['pitch_spectrogram']
                # print(f"pitch-config: {pitch_config}")
                pitch_spec, f0, f0_mean, f0_std, t = calc_pitch_spectrogram(
                    filepath=audio_path,
                    config=pitch_config
                )
                pitch_spec_path = pitch_feature_dir / f'pitch_spec_{fname}.npy'
                _ = np.save(pitch_spec_path, pitch_spec)
                pitch_contour_path = pitch_feature_dir / f'pitch_contour_{fname}.npy'
                _ = np.save(pitch_contour_path, f0)
                pitch_contour_mean_path = pitch_feature_dir / f'pitch_contour_mean_{fname}.npy'
                _ = np.save(pitch_contour_mean_path, f0_mean)
                pitch_contour_std_path = pitch_feature_dir / f'pitch_contour_std_{fname}.npy'
                _ = np.save(pitch_contour_std_path, f0_std)
                
                # Energy Features
                energy_config = config['preprocessing']['energy']
                frame_energy = calculate_stft_energy(
                    filepath=audio_path,
                    config=energy_config
                )
                energy_path = energy_feature_dir / f'energy_{fname}.npy'
                _ = np.save(energy_path, frame_energy)

            else:
                print(f"File not available for: {fname}")
