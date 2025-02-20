import torch
import torch.nn as nn
import lightning as l
from .zsn2n_optimizer import ZSN2NOptimizer
from .zsn2n_trainer import ZSN2NTrainer
from .network.CConv import CConv2d, CConvTranspose2d


class ZSN2N(ZSN2NTrainer, ZSN2NOptimizer, l.LightningModule):

    def __init__(self, config):
        # LightningModuleの初期化を先に呼び出す
        l.LightningModule.__init__(self)

        # 他のクラスの__init__も引数付きで呼び出す
        ZSN2NTrainer.__init__(self, config=config)
        ZSN2NOptimizer.__init__(self, config=config)

        self.input_channels = config.model.input_channels
        self.embed_dim = config.model.embed_dim
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.downsample1 = CConv2d(
            self.input_channels, self.embed_dim, kernel_size=3, stride=2, padding=1
        )  # 出力サイズ: /2
        self.downsample2 = CConv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)  # 出力サイズ: /4
        self.downsample3 = CConv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)  # 出力サイズ: /8
        self.downsample4 = CConv2d(
            self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1
        )  # 出力サイズ: /16

        # デコーダ
        self.upsample1 = CConvTranspose2d(
            self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 出力サイズ: /4
        self.upsample2 = CConvTranspose2d(
            self.embed_dim * 2, self.embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 出力サイズ: /2
        self.upsample3 = CConvTranspose2d(
            self.embed_dim * 2, self.embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 出力サイズ: 元のサイズ

        self.upsample4 = CConvTranspose2d(
            self.embed_dim * 2, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # 活性化関数
        self.act = nn.ReLU()

    def forward(self, x):
        # 入力の形状を確認
        # print(f"Input shape: {x.shape}")  # 例: [1, 2, 512, 256]

        # エンコーダ
        d1 = self.act(self.downsample1(x))
        # print(f"d1.shape : {d1.shape}")  # 例: [1, embed_dim, 256, 128]
        d2 = self.act(self.downsample2(d1))
        # print(f"d2.shape : {d2.shape}")  # 例: [1, embed_dim, 128, 64]
        d3 = self.act(self.downsample3(d2))
        # print(f"d3.shape : {d3.shape}")  # 例: [1, embed_dim, 64, 32]
        d4 = self.act(self.downsample4(d3))

        # デコーダ
        u1 = self.upsample1(d4)
        # print(f"u1.shape : {u1.shape}")  # 例: [1, embed_dim, 128, 64]

        # スキップ接続
        c0 = torch.cat((u1, d3), dim=0)
        # print(f"c0.shape : {c0.shape}")  # 例: [1, 2*embed_dim, 128, 64]

        u2 = self.upsample2(c0)
        # print(f"u2.shape : {u2.shape}")  # 例: [1, embed_dim, 256, 128]

        # スキップ接続
        c1 = torch.cat((u2, d2), dim=0)
        # print(f"c1.shape : {c1.shape}")  # 例: [1, 2*embed_dim, 256, 128]

        u3 = self.upsample3(c1)
        # print(f"u3.shape : {u3.shape}")  # 例: [1, input_channels, 512, 256]
        c2 = torch.cat((u3, d1), dim=0)

        u4 = self.upsample4(c2)

        return u4 * x
