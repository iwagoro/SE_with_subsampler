import torch
import librosa
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    DeepNoiseSuppressionMeanOpinionScore,
    SignalNoiseRatio,
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
)


class Metrics:
    def __init__(self, model, clean, pred):
        # モデルインスタンスから直接属性を参照
        self.clean = clean
        self.pred = pred
        self.n_fft = model.n_fft
        self.hop_length = model.hop_length
        self.sr = model.sample_rate
        self.resr = 16000

        # 16kHz にリサンプリング
        self.resampled_clean = self._resample(self.clean)
        self.resampled_pred = self._resample(self.pred)

        # メトリックオブジェクトを事前に生成
        self.pesq_fn = PerceptualEvaluationSpeechQuality(self.resr, mode="wb")
        self.stoi_fn = ShortTimeObjectiveIntelligibility(self.resr, False)
        self.snr_fn = SignalNoiseRatio()
        self.si_snr_fn = ScaleInvariantSignalDistortionRatio()
        self.sdr_fn = SignalDistortionRatio()
        self.si_sdr_fn = ScaleInvariantSignalDistortionRatio()

    def get_metrics(self):
        return {
            "pesq": self.pesq(),
            "stoi": self.stoi(),
            "snr": self.snr(),
            "si_snr": self.si_snr(),
            "sdr": self.sdr(),
            "si_sdr": self.si_sdr(),
        }

    def _resample(self, wav):
        if self.sr != self.resr:
            wav = librosa.resample(y=wav.cpu().detach().numpy(), orig_sr=self.sr, target_sr=self.resr)
            wav = torch.tensor(wav)
        return wav

    def pesq(self):
        return self.pesq_fn(self.resampled_pred, self.resampled_clean)

    def stoi(self):
        return self.stoi_fn(self.resampled_pred, self.resampled_clean)

    def snr(self):
        return self.snr_fn(self.pred, self.clean)

    def si_snr(self):
        return self.si_snr_fn(self.pred, self.clean)

    def sdr(self):
        return self.sdr_fn(self.pred, self.clean)

    def si_sdr(self):
        return self.si_sdr_fn(self.pred, self.clean)
