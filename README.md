# ディレクトリの説明

## config：学習を回す時の設定用

-   config.yaml：学習の設定ファイル．yaml の書き方 →https://qiita.com/supercell1023koeda/items/6b1fd399b1957077651a

## data：データセット用

-   source/urban：UrbanSound8k のノイズのデータセット
-   source/clean：クリーン音声のデータセット

## data_collection：論文とかのデータのプロット用

-   modules/dataset：データセットを読み込むプログラム
-   modules/down_sample：音声をダウンサンプルするプログラム
-   modules/low_pass：ローパスフィルターのプログラム
-   modules/plot_spectrogram：スペクトログラムを表示するプログラム
-   modules/subsample：サブサンプラーのプログラム
-   modules/zncc：スペクトログラムの相互相関マップを作成するプログラム

-   plot_enhanced_spectrogram.ipynb：noisy,clean,pred をプロットするプログラム
-   plot_subsample_spectrogram.ipynb：noisy と，サブサンプルした後の noisy ペアをプロットするプログラム
-   plot_zncc_map.ipynb：相互相関マップをプロットするプログラム
-   wandb.ipynb：wandb に保存した学習結果から，データを収集するプログラム

## logs：学習結果をまとめる用

-   使うことないからほっといていいかも

## src：プログラム等

-   data/dataset.py：データセットをモデルに提供するプログラム
-   models/network/CConv：スペクトログラムを実部・虚部ごとに別々に畳み込むプログラム
-   models/zsn2n_optimizer.py：オプティマイザとスケジューラ関連
-   models/zsn2n_trainer.py：学習時のロジック関連
-   zsn2n.py：ネットワーク関連
-   utils/metrics.py：PESQ とか SDR とか評価指標計算関連
-   utils/save_wav.py：音声を wandb に保存
-   utils/subsample.py：音声のサブサンプル関連
-   train.py：メインのプログラムファイル．
