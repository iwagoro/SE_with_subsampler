import hydra
from omegaconf import DictConfig, OmegaConf
from itertools import product
from data.dataset import SpeechDataset
from models.zsn2n import ZSN2N
import glob
import os
import lightning as l
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, MLFlowLogger
import wandb


def create_combinations(cfg: DictConfig):

    # whiteの場合はから配列，urbanの場合はファイルパスのリストを取得
    noise_file_path_list = sorted(glob.glob(os.path.join(cfg.dataset.noise_dir_path, cfg.dataset.noise_type, "*.wav")))[
        : cfg.dataset.noise_samples
    ]
    clean_file_path_list = sorted(glob.glob(os.path.join(cfg.dataset.clean_dir_path, "*.wav")))[
        : cfg.dataset.clean_samples
    ]

    if noise_file_path_list == []:
        cfg.dataset.noise_file_path = ["" for _ in range(cfg.dataset.noise_samples)]
    else:
        cfg.dataset.noise_file_path = noise_file_path_list

    if clean_file_path_list == []:
        cfg.dataset.clean_file_path = ["" for _ in range(cfg.dataset.clean_samples)]
    else:
        cfg.dataset.clean_file_path = clean_file_path_list

    param_lists = {}
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):  # 配列になっているパラメータを検出
                    param_lists[f"{key}.{sub_key}"] = sub_value

    # 組み合わせを生成
    param_combinations = list(product(*param_lists.values()))
    param_keys = list(param_lists.keys())

    new_cfgs = []

    for combination in param_combinations:
        # cfgをコピーしてパラメータを更新
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        cfg_copy = OmegaConf.create(cfg_copy)

        # 組み合わせをcfg_copyに適用
        for i, key in enumerate(param_keys):
            # ネストされたキーを動的に適用
            OmegaConf.update(cfg_copy, key, combination[i])

        new_cfgs.append(cfg_copy)

    print(f"Number of combinations: {len(new_cfgs)}")

    return new_cfgs


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    new_cfgs = create_combinations(cfg)

    for i, cfg in enumerate(new_cfgs):

        # slice path name with "/" and get the last element
        # print(f"dataset_number : {cfg.dataset.clean_file_path.split('/')[-1]}")
        dataset = SpeechDataset(cfg)
        model = ZSN2N(cfg)
        accelerator = "cuda"
        devices = [1]

        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green_yellow",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="cyan",
                processing_speed="#ff1493",
                metrics="#ff1493",
                metrics_text_delimiter="\n",
            )
        )

        early_stopping = EarlyStopping(
            monitor="train_loss",
            patience=cfg.training.early_stopping.patience,
            mode="min",
            # verbose=True,
        )

        tags = [
            f"snr_level: {cfg.dataset.snr_level}",
            f"sample_rate: {cfg.dataset.sample_rate}",
            f"max_len: {cfg.dataset.max_len}",
            f"n_fft: {cfg.dataset.n_fft}",
            f"hop_length: {cfg.dataset.hop_length}",
            f"embed_dim: {cfg.model.embed_dim}",
            f"subsample_k: {cfg.model.subsample_k}",
            f"loss_type: {cfg.model.loss_type}",
            f"clean_file: {cfg.dataset.clean_file_path.split('/')[-1]}",
            f"noise_file: {cfg.dataset.noise_file_path.split('/')[-1]}"
        ]

        # MLflowLoggerの設定
        logger = WandbLogger(
            project=cfg.dataset.noise_type+"_"+cfg.model.loss_type,
            name=f"{cfg.train.keyword}_{cfg.model.subsample_k}_{cfg.model.embed_dim}_{cfg.dataset.clean_file_path.split('/')[-1]}",
            save_dir=cfg.project.log_dir,
            offline=True,
            # offline=False,
            reinit=True,
            tags=tags,
        )

        trainer = l.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=cfg.training.max_epoch,
            callbacks=[progress_bar, early_stopping],
            # callbacks=[progress_bar],
            logger=logger,
            log_every_n_steps=1,
            default_root_dir=cfg.project.log_dir,
        )

        data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size)

        # トレーニングの実行
        # trainer.fit(model, train_dataloaders=data_loader, val_dataloaders=data_loader)
        trainer.fit(model, data_loader)

        # 予測の実行
        trainer.predict(model, data_loader)
        
        wandb.finish()


if __name__ == "__main__":
    main()
