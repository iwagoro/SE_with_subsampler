import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display

from PIL import Image
from io import BytesIO


def show_spec_and_save_pdf(wav, title, sample_rate):
    hop_length = 256
    n_fft = 1024
    sample_rate = sample_rate
    min_db = -90
    max_db = 0

    # スペクトログラムの作成
    fig, ax = plt.subplots()
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(
        D_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="jet",
        vmin=min_db,
        vmax=max_db,
        ax=ax,
    )
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_title(title, fontsize=14)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # 一時ファイルにPDFとして保存
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    plt.close(fig)
    image = Image.open(buf)
    return image


def save_wav_and_spec(logger, sr, wav_obj):
    for wav_name, wav in wav_obj.items():
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        wav = np.squeeze(wav).astype(np.float32)

        # 音声データのログ
        logger.experiment.log(
            {
                f"{wav_name}": wandb.Audio(wav, sample_rate=sr, caption=f"{wav_name}"),
            }
        )

        # スペクトログラムをPDFとして保存し、アーティファクトとしてログ
        logger.experiment.log(
            {
                f"{wav_name}_spec": wandb.Image(
                    show_spec_and_save_pdf(wav, title=f"{wav_name} spectrogram", sample_rate=sr)
                )
            },
        )
