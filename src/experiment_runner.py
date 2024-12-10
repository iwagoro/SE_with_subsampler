import os
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="experiment_runner_config", version_base=None)
def main(cfg: DictConfig):
    # CleanおよびNoiseディレクトリからファイルを取得してソート
    clean_files = sorted(os.listdir(cfg.dataset.clean_path))[: cfg.dataset.clean_samples]
    noise_files = (
        [""]
        if cfg.dataset.noise_type == "white"
        else sorted(os.listdir(os.path.join(cfg.dataset.noise_path, cfg.dataset.noise_type)))[
            : cfg.dataset.noise_samples
        ]
    )

    # ディレクトリパスをファイルパスに変換
    clean_files = [os.path.join(cfg.dataset.clean_path, f) for f in clean_files]
    noise_files = (
        [""]
        if cfg.dataset.noise_type == "white"
        else [os.path.join(cfg.dataset.noise_path, cfg.dataset.noise_type, f) for f in noise_files]
    )

    # 各パラメータの組み合わせでループ
    for embed_dim in cfg.model.embed_dim:
        for subsample_k in cfg.model.subsample_k:
            for ss_num_frames in cfg.dataset.ss_num_frames:
                for snr_level in cfg.dataset.snr_levels:
                    for clean_file in clean_files:
                        for noise_file in noise_files:
                            for noisy_levels in cfg.dataset.noisy_levels:
                                # Hydraの設定をオーバーライドするコマンドを作成
                                command = [
                                    "/Users/rockwell/miniforge3/envs/ml/bin/python",
                                    "src/train.py",
                                    f"dataset.noisy_level={noisy_levels}",
                                    f"train.keyword={cfg.train.keyword}",
                                    f"dataset.noise_type={cfg.dataset.noise_type}",
                                    f"train.clean_file_path={clean_file}",
                                    f"train.noise_file_path={noise_file}",
                                    f"train.snr_level={snr_level}",
                                    f"model.embed_dim={embed_dim}",
                                    f"model.subsample_k={subsample_k}",
                                    f"dataset.ss_num_frames={ss_num_frames}",
                                ]

                                # コマンドを実行
                                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
