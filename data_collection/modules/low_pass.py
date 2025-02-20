import torch
from scipy.signal import butter, lfilter


def apply_lowpass_filter(waveform):
    cut_order = 5
    cut_off = 16000 / 8
    sr = 16000
    b, a = butter(cut_order, cut_off, btype="low", fs=sr)
    wav = lfilter(b, a, waveform.squeeze(0).cpu().detach().numpy())
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    return wav
