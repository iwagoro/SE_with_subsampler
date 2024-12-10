
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa


class SpeechDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.noise_file_path = config.dataset.noise_file_path
        self.clean_file_path = config.dataset.clean_file_path
        self.snr_level = config.dataset.snr_level
        self.max_len = config.dataset.max_len
        self.sr = config.dataset.sample_rate
        self.batch_size = config.training.batch_size
        self.noise_type = config.dataset.noise_type

    def __len__(self):
        return self.batch_size

    def load_sample(self, file):
        waveform, osr = torchaudio.load(file)
        # サンプリングレートが指定されている場合はリサンプリング
        if self.sr != osr:
            waveform = librosa.resample(y=waveform.cpu().detach().numpy(), orig_sr=osr, target_sr=self.sr)
            waveform = torch.tensor(waveform)
        # モノラルに変換
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def __getitem__(self, index):
        clean = self.load_sample(self.clean_file_path)
        noise = None
        if self.noise_type != "white":
            noise = self.load_sample(self.noise_file_path)

        # サンプルの整形
        clean = self._prepare_sample(clean)
        noisy = None
        if self.noise_type == "white":
            noisy = self.add_white_noise(clean, self.snr_level)
        else:
            noisy = self.add_noise(clean, noise, self.snr_level)

        # noisy = self.load_sample(
        #     "/Users/rockwell/Documents/python/SE-with-SubSampler/logs/wandb/wandb/offline-run-20241118_195028-t9l83pfo/files/media/audio/pred_clean_202_c71bdc389c1e91bfca6e.wav"
        # )

        return noisy, clean

    def _prepare_sample(self, waveform):
        channels, current_len = waveform.shape
        output = torch.zeros((channels, self.max_len), dtype=torch.float32, device=waveform.device)
        output[:, -min(current_len, self.max_len) :] = waveform[:, : min(current_len, self.max_len)]
        return output

    def add_white_noise(self, clean_waveform, snr):
        clean_power = torch.mean(clean_waveform**2)
        noise_power = clean_power / (10 ** (snr / 10))
        white_noise = torch.randn_like(clean_waveform) * torch.sqrt(noise_power)
        noisy_waveform = clean_waveform + white_noise
        return self._prepare_sample(noisy_waveform)

    def add_noise(self, clean_waveform, noise_waveform, snr):
        clean_len = clean_waveform.shape[1]
        noise_len = noise_waveform.shape[1]

        if noise_len < clean_len:
            repeat_factor = (clean_len // noise_len) + 1
            noise_waveform = noise_waveform.repeat(1, repeat_factor)[:, :clean_len]
        else:
            noise_waveform = noise_waveform[:, :clean_len]

        clean_power = torch.mean(clean_waveform**2)
        noise_power = torch.mean(noise_waveform**2)
        scaling_factor = torch.sqrt(clean_power / (noise_power * 10 ** (snr / 10)))
        scaled_noise = noise_waveform * scaling_factor
        noisy_waveform = clean_waveform + scaled_noise

        return self._prepare_sample(noisy_waveform)
