import torch
import librosa
import numpy as np


def ss(wav, n_fft, hop_length, num_frame=10):
    # TensorをNumPy配列に変換
    wav = wav.squeeze(0).squeeze(0).cpu().detach().numpy()

    # STFTを計算してスペクトルを取得
    spectrum = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)

    # 振幅スペクトルと位相スペクトルを計算
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # ノイズスペクトルを推定（最初のnum_frameフレームの平均）
    noise_spectrum = np.mean(magnitude[:, :num_frame], axis=1, keepdims=True)

    # ゲインを計算（振幅スペクトル - ノイズスペクトル）/ 振幅スペクトル
    # 振幅スペクトルが0の場合のゼロ割りを防ぐために、εを加算
    epsilon = 1e-8
    gain = (magnitude - noise_spectrum) / (magnitude + epsilon)
    gain = np.clip(gain, 0, 1)  # ゲインを0から1の範囲に制限

    # ノイズ除去後のスペクトルを計算
    enhanced_spectrum = gain * magnitude * np.exp(1j * phase)

    # 逆STFTで時間領域信号を再構成
    enhanced_wav = librosa.istft(enhanced_spectrum, hop_length=hop_length, length=len(wav))

    # NumPy配列をTensorに変換
    enhanced_wav = torch.from_numpy(enhanced_wav).unsqueeze(0).unsqueeze(0)

    # print(enhanced_wav.shape)

    return enhanced_wav
