import numpy as np
import librosa


def compute_zncc_map(g1, g2, sr=8000, n_fft=1022, hop_length=32, patch_size=7, padding_mode="reflect", eps=1e-10):
    """
    ZNCC（ゼロ平均正規化相互相関）マップを計算する関数

    パラメータ:
    g1, g2 (np.ndarray): 比較する2つの音声信号
    sr (int): サンプリングレート
    n_fft (int): STFTのフレームサイズ
    hop_length (int): STFTのホップサイズ
    patch_size (int): パッチサイズ（奇数）
    padding_mode (str): パディングモード
    eps (float): ゼロ除算防止用の小さい値

    戻り値:
    np.ndarray: ZNCCマップ
    """
    # 入力チェック
    if patch_size % 2 == 0:
        raise ValueError("パッチサイズは奇数でなければなりません")

    # STFT計算
    D1 = librosa.stft(g1, n_fft=n_fft, hop_length=hop_length)
    D2 = librosa.stft(g2, n_fft=n_fft, hop_length=hop_length)

    # 振幅スペクトル取得
    D1_mag = np.abs(D1)
    D2_mag = np.abs(D2)

    # D1_mag と D2_mag のサイズを合わせる
    min_freq_bins = min(D1_mag.shape[0], D2_mag.shape[0])
    min_time_frames = min(D1_mag.shape[1], D2_mag.shape[1])
    D1_mag = D1_mag[:min_freq_bins, :min_time_frames]
    D2_mag = D2_mag[:min_freq_bins, :min_time_frames]

    # パッチサイズを奇数の正方形に設定
    if patch_size % 2 == 0:
        raise ValueError("パッチサイズは奇数でなければなりません。")
    patch_freq_size = patch_size
    patch_time_size = patch_size

    # パディングサイズの計算
    pad_freq = patch_freq_size // 2
    pad_time = patch_time_size // 2

    # D1_mag と D2_mag にパディングを追加
    D1_padded = np.pad(D1_mag, ((pad_freq, pad_freq), (pad_time, pad_time)), mode=padding_mode)
    D2_padded = np.pad(D2_mag, ((pad_freq, pad_freq), (pad_time, pad_time)), mode=padding_mode)

    # 元の画像サイズ
    original_freq_bins, original_time_frames = D1_mag.shape

    # 相互相関マップを元の画像と同じサイズで初期化
    corr_map = np.zeros((original_freq_bins, original_time_frames))

    # 各ピクセル位置に対してパッチを抽出してZNCCを計算
    for i in range(original_freq_bins):
        for j in range(original_time_frames):
            # パッチの開始位置
            f_start = i
            f_end = i + patch_freq_size
            t_start = j
            t_end = j + patch_time_size

            patch1 = D1_padded[f_start:f_end, t_start:t_end]
            patch2 = D2_padded[f_start:f_end, t_start:t_end]

            p1_flat = patch1.flatten()
            p2_flat = patch2.flatten()

            # ZNCCの計算
            mean_x = np.mean(p1_flat)
            mean_y = np.mean(p2_flat)
            numerator = np.sum((p1_flat - mean_x) * (p2_flat - mean_y))
            denominator = np.sqrt(np.sum((p1_flat - mean_x) ** 2) * np.sum((p2_flat - mean_y) ** 2)) + eps
            zncc = numerator / denominator

            corr_map[i, j] = zncc

    return corr_map
