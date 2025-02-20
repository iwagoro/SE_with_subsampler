import torch
import torchaudio


def downsample_to_8k(wav, sr):
    """
    音声信号を指定されたサンプルレートから8kHzにダウンサンプリングします。

    Parameters:
    - wav: 音声信号のnumpy配列
    - sr: 元のサンプルレート

    Returns:
    - ダウンサンプリングされた音声信号のnumpy配列
    """
    wav = torch.tensor(wav)
    wav = wav.unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)
    wav = resampler(wav)
    return wav[0].numpy()
