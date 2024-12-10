import torch
from utils.subsample import subsample
from utils.save_wav import save_wav_and_spec
from utils.spectarl_subtraction import ss
from scipy.signal import butter, lfilter
from utils.metrics import Metrics
import wandb
import torch.nn.functional as F

def wsdr_loss( x, y_pred, y_true, eps=1e-8):  
        y_pred = y_pred.flatten(1)
        y_true = y_true.flatten(1)
        x = x.flatten(1)

        def sdr_fn(true, pred, eps=1e-8):
            num = torch.sum(true * pred, dim=1)
            den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
            return -(num / (den + eps))

        # true and estimated noise
        z_true = x - y_true
        z_pred = x - y_pred

        a = torch.sum(y_true ** 2, dim=1) / (torch.sum(y_true ** 2, dim=1) + torch.sum(z_true ** 2, dim=1) + eps)
        wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
        return torch.mean(wSDR)
    
class ZSN2NTrainer:
    def __init__(self, config):
        self.config = config
        self.n_fft = config.dataset.n_fft
        self.hop_length = config.dataset.hop_length
        self.sample_rate = config.dataset.sample_rate
        self.sr = config.dataset.sample_rate
        self.cut_off = self.sr / 4
        self.cur_order = 5
        self.config.noise_duration = 0.6
        self.loss_type = config.model.loss_type
        # # 偶数のみの範囲を作成
        # even_numbers = torch.arange(2, config.model.subsample_k + 1, step=2)
        # self.subsample_k = even_numbers[torch.randint(0, len(even_numbers), (1,))].item()

        self.subsample_k = config.model.subsample_k

    def _apply_lowpass_filter(self, waveform):
        b, a = butter(self.cur_order, self.cut_off, btype="low", fs=self.sr)
        wav = lfilter(b, a, waveform.squeeze(0).cpu().detach().numpy())
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        return wav

    def training_step(self, batch, batch_idx):
        noisy, _ = batch
        noisy = self._apply_lowpass_filter(noisy).to(self.device)
        # noisy += self.noisy_level + torch.rand_like(noisy)

        if self.loss_type == "stft":
            window = torch.hann_window(self.n_fft,device=noisy.device)
            g1, g2 = subsample(noisy, self.subsample_k)
            g1_stft = torch.stft(g1.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True,window=window)
            g1_stft = torch.view_as_real(g1_stft)

            g2_stft = torch.stft(g2.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True,window=window)
            g2_stft = torch.view_as_real(g2_stft)

            # pred1_stft = g1_stft - self(g1_stft)
            pred1_stft = self(g1_stft)
            # pred1 = torch.view_as_complex(pred1_stft)
            # pred1 = torch.istft(pred1, n_fft=self.n_fft, hop_length=self.hop_length, window=window, length=g1.size(-1)).unsqueeze(0)
            # pred2_stft = g2_stft - self(g2_stft)
            pred2_stft =  self(g2_stft)
            # pred2 = torch.view_as_complex(pred2_stft)
            # pred2 = torch.istft(pred2, n_fft=self.n_fft, hop_length=self.hop_length, window=window, length=g2.size(-1)).unsqueeze(0)

            loss_res = 0.5 * (torch.nn.MSELoss()(g1_stft, pred2_stft) + torch.nn.MSELoss()(g2_stft, pred1_stft))
            # loss_res_time = 0.5 * (torch.nn.MSELoss()(g1, pred2) + torch.nn.MSELoss()(g2, pred1))

            noisy_stft = torch.stft(noisy.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True,window=window)
            noisy_stft = torch.view_as_real(noisy_stft)

            # denoised_stft = noisy_stft - self(noisy_stft)
            denoised_stft = self(noisy_stft)
            denoised = torch.view_as_complex(denoised_stft)
            denoised = torch.istft(denoised, n_fft=self.n_fft, hop_length=self.hop_length,window=window)

            dg1, dg2 = subsample(denoised.unsqueeze(0), self.subsample_k)
            dg1_stft = torch.stft(dg1.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True,window=window)
            dg1_stft = torch.view_as_real(dg1_stft)

            dg2_stft = torch.stft(dg2.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True,window=window)
            dg2_stft = torch.view_as_real(dg2_stft)

            loss_cons = 0.5 * (torch.nn.MSELoss()(pred1_stft, dg1_stft) + torch.nn.MSELoss()(pred2_stft, dg2_stft))
            # loss_cons_time = 0.5 * (torch.nn.MSELoss()(pred1, dg1) + torch.nn.MSELoss()(pred2, dg2))
            loss = (loss_res + loss_cons) #+ 0*(loss_res_time + loss_cons_time)

            self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=True)
            return loss

    def predict_step(self, batch, batch_idx):
        noisy, clean = batch
        if self.loss_type == "stft":
            window = torch.hann_window(self.n_fft,device=noisy.device)
            noisy_stft = torch.stft(noisy.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True,window=window)
            noisy_stft = torch.view_as_real(noisy_stft)

            # pred_clean_stft = noisy_stft - self(noisy_stft)
            # pred_noise_stft = self(noisy_stft)
            pred_clean_stft = self(noisy_stft)
            pred_noise_stft = noisy_stft - self(noisy_stft)

            pred_clean = torch.view_as_complex(pred_clean_stft)
            pred_clean = torch.istft(pred_clean, n_fft=self.n_fft, hop_length=self.hop_length,window=window)

            pred_noise = torch.view_as_complex(pred_noise_stft)
            pred_noise = torch.istft(pred_noise, n_fft=self.n_fft, hop_length=self.hop_length,window=window)

            noisy = noisy.detach().cpu()[0]
            pred_clean = pred_clean.detach().cpu()
            pred_noise = pred_noise.detach().cpu()
            clean = clean.detach().cpu()[0]


            wav_obj = {
                "noisy": noisy,
                "pred_clean": pred_clean,
                "pred_noise": pred_noise,
                "clean": clean,
            }

            save_wav_and_spec(self.logger, self.sample_rate, wav_obj)

            noisy_metrics = Metrics(self, clean, noisy).get_metrics()
            # for key, value in noisy_metrics.items():
            #     self.log(key, value)
            pred_clean_metrics = Metrics(self, clean, pred_clean).get_metrics()
            # for key, value in pred_clean_metrics.items():
            #     self.log(key, value)
            pred_noise_metrics = Metrics(self, clean, pred_noise).get_metrics()

            wandb.log({"noisy_metrics": noisy_metrics})
            wandb.log({"pred_clean_metrics": pred_clean_metrics})
            wandb.log({"pred_noise_metrics": pred_noise_metrics})

            print(f"Noisy: {noisy_metrics}")
            print(f"Pred Clean: {pred_clean_metrics}")
            print(f"Pred Noise: {pred_noise_metrics}")
