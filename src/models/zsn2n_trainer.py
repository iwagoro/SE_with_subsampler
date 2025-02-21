import torch
from utils.subsample import subsample
from utils.save_wav import save_wav_and_spec
from scipy.signal import butter, lfilter
from utils.metrics import Metrics
import wandb
import torch.nn.functional as F


class ZSN2NTrainer:
    def __init__(self, config):
        self.config = config
        self.n_fft = config.dataset.n_fft
        self.hop_length = config.dataset.hop_length
        self.sample_rate = config.dataset.sample_rate
        self.sr = config.dataset.sample_rate
        self.is_lp = config.model.is_lp
        self.cut_off = self.sr / 4
        self.cur_order = 5

        self.subsample_k = config.model.subsample_k

    def _apply_lowpass_filter(self, waveform):
        b, a = butter(self.cur_order, self.cut_off, btype="low", fs=self.sr)
        wav = lfilter(b, a, waveform.squeeze(0).cpu().detach().numpy())
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        return wav

    def training_step(self, batch, batch_idx):
        noisy, _ = batch
        if self.is_lp:
            noisy = self._apply_lowpass_filter(noisy).to(self.device)

        window = torch.hann_window(self.n_fft, device=noisy.device)

        #! Subsample
        g1, g2 = subsample(noisy, self.subsample_k)

        g1_stft = torch.stft(
            g1.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window
        )
        g1_stft = torch.view_as_real(g1_stft)

        g2_stft = torch.stft(
            g2.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window
        )
        g2_stft = torch.view_as_real(g2_stft)

        #! Estimate the noise from subsampled signals
        pred1_stft = g1_stft - self(g1_stft)
        pred2_stft = g2_stft - self(g2_stft)

        #! Loss
        loss_res = 0.5 * (torch.nn.MSELoss()(g1_stft, pred2_stft) + torch.nn.MSELoss()(g2_stft, pred1_stft))

        #! Estimate the noise from the original signal
        noisy_stft = torch.stft(
            noisy.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window
        )
        noisy_stft = torch.view_as_real(noisy_stft)
        denoised_stft = noisy_stft - self(noisy_stft)
        denoised = torch.view_as_complex(denoised_stft)
        denoised = torch.istft(denoised, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        dg1, dg2 = subsample(denoised.unsqueeze(0), self.subsample_k)
        dg1_stft = torch.stft(
            dg1.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window
        )
        dg1_stft = torch.view_as_real(dg1_stft)
        dg2_stft = torch.stft(
            dg2.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window
        )
        dg2_stft = torch.view_as_real(dg2_stft)

        #! Loss
        loss_cons = 0.5 * (torch.nn.MSELoss()(pred1_stft, dg1_stft) + torch.nn.MSELoss()(pred2_stft, dg2_stft))
        loss = loss_res + loss_cons

        #! Logging
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=True)
        return loss

    def predict_step(self, batch, batch_idx):
        noisy, clean = batch
        window = torch.hann_window(self.n_fft, device=noisy.device)
        noisy_stft = torch.stft(
            noisy.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=window
        )
        noisy_stft = torch.view_as_real(noisy_stft)
        pred_clean_stft = noisy_stft - self(noisy_stft)
        pred_noise_stft = self(noisy_stft)
        pred_clean = torch.view_as_complex(pred_clean_stft)
        pred_clean = torch.istft(pred_clean, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        pred_noise = torch.view_as_complex(pred_noise_stft)
        pred_noise = torch.istft(pred_noise, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        noisy = noisy.detach().cpu()[0]
        pred_clean = pred_clean.detach().cpu()[0]
        pred_noise = pred_noise.detach().cpu()[0]
        clean = clean.detach().cpu()[0]
        wav_obj = {
            "noisy": noisy,
            "pred_clean": pred_clean,
            "pred_noise": pred_noise,
            "clean": clean,
        }
        save_wav_and_spec(self.logger, self.sample_rate, wav_obj)
        noisy_metrics = Metrics(self, clean, noisy).get_metrics()
        pred_clean_metrics = Metrics(self, clean, pred_clean).get_metrics()
        pred_noise_metrics = Metrics(self, clean, pred_noise).get_metrics()
        wandb.log({"noisy_metrics": noisy_metrics})
        wandb.log({"pred_clean_metrics": pred_clean_metrics})
        wandb.log({"pred_noise_metrics": pred_noise_metrics})
        print(f"Noisy: {noisy_metrics}")
        print(f"Pred Clean: {pred_clean_metrics}")
        print(f"Pred Noise: {pred_noise_metrics}")
