# import torch
# import torch.nn as nn


# class CConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super(CConv2d, self).__init__()

#         # 実部用のdepth-wiseとpoint-wiseの畳み込み
#         self.depthwise_conv_real = nn.Conv2d(
#             in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels
#         )
#         self.pointwise_conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#         # 虚部用のdepth-wiseとpoint-wiseの畳み込み
#         self.depthwise_conv_imag = nn.Conv2d(
#             in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels
#         )
#         self.pointwise_conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         # 実部と虚部を分割
#         x_real = x[..., 0]
#         x_im = x[..., 1]

#         # depth-wiseとpoint-wiseの順で畳み込みを実行
#         c_real = self.pointwise_conv_real(self.depthwise_conv_real(x_real)) - self.pointwise_conv_imag(
#             self.depthwise_conv_imag(x_im)
#         )
#         c_im = self.pointwise_conv_imag(self.depthwise_conv_real(x_im)) + self.pointwise_conv_real(
#             self.depthwise_conv_real(x_im)
#         )

#         # 実部と虚部を再度組み合わせて出力
#         output = torch.stack([c_real, c_im], dim=-1)
#         return output


# class CConvTranspose2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
#         super(CConvTranspose2d, self).__init__()

#         # 実部用のdepth-wiseとpoint-wiseの逆畳み込み
#         self.depthwise_conv_transposed_real = nn.ConvTranspose2d(
#             in_channels,
#             in_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             output_padding=output_padding,
#             groups=in_channels,
#         )
#         self.pointwise_conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#         # 虚部用のdepth-wiseとpoint-wiseの逆畳み込み
#         self.depthwise_conv_transposed_imag = nn.ConvTranspose2d(
#             in_channels,
#             in_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             output_padding=output_padding,
#             groups=in_channels,
#         )
#         self.pointwise_conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         # 実部と虚部を分割
#         x_real = x[..., 0]
#         x_im = x[..., 1]

#         # depth-wiseとpoint-wiseの順で逆畳み込みを実行
#         c_real = self.pointwise_conv_real(self.depthwise_conv_transposed_real(x_real)) - self.pointwise_conv_imag(
#             self.depthwise_conv_transposed_imag(x_im)
#         )
#         c_im = self.pointwise_conv_imag(self.depthwise_conv_transposed_real(x_im)) + self.pointwise_conv_real(
#             self.depthwise_conv_transposed_real(x_real)
#         )

#         # 実部と虚部を再度組み合わせて出力
#         output = torch.stack([c_real, c_im], dim=-1)
#         return output
import torch.nn as nn
import torch


class CConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dtype=torch.float32,
            stride=self.stride,
            dilation=dilation,
        )

        self.im_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dtype=torch.float32,
            stride=self.stride,
            dilation=dilation,
        )

        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)

        x_real = x[..., 0]
        x_im = x[..., 1]
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.stack([c_real, c_im], dim=-1)
        return output


class CConvTranspose2d(nn.Module):
    """
    Class of complex valued dilation convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0, dilation=1):
        super().__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride

        self.real_convt = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            output_padding=self.output_padding,
            padding=self.padding,
            dtype=torch.float32,
            stride=self.stride,
            dilation=dilation,
        )

        self.im_convt = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            output_padding=self.output_padding,
            padding=self.padding,
            dtype=torch.float32,
            stride=self.stride,
            dilation=dilation,
        )

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)

        x = x.to(torch.float32)

        x_real = x[..., 0]
        x_im = x[..., 1]

        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)

        output = torch.stack([ct_real, ct_im], dim=-1)
        return output
