import numpy as np
import librosa
import librosa.display


def plot_spectrogram(ax, wav, sample_rate, title, font_size=32):
    """
    スペクトログラムをプロットし、タイトルを下部中央に配置します。
    """
    hop_length = 64
    n_fft = 1022
    min_db = -100
    max_db = 0

    # STFTの計算
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # スペクトログラムの表示
    img = librosa.display.specshow(
        D_db,
        sr=sample_rate,
        hop_length=hop_length,
        cmap="jet",
        vmin=min_db,
        vmax=max_db,
        ax=ax,
    )

    # タイトルを下部中央に配置（日本語OK）
    ax.axis("off")  # 軸を非表示
    ax.text(
        0.5,
        -0.05,
        title,
        fontsize=font_size,
        ha="center",
        va="top",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )
    return img
