import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, reduced_dim):
        super().__init__()
        self.squeeze_and_excitation_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # C x H X W -> C x 1 x 1, i.e. one value as output
            nn.Conv2d(input_channels, reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, input_channels, kernel_size=1),  # restoring size to input size
            nn.Sigmoid()
        )

    def forward(self, x):
        # "how much is each channel prioritized"
        return x * self.squeeze_and_excitation_module(x)


class FFMMNet(nn.Module):
    def __init__(self, dropout=0.3, input_channels=4, output_channel=1, num_layers=7):
        super(FFMMNet, self).__init__()

        self.conv_layers = nn.ModuleList()
        _input_channels_list = [input_channels, 8, 16, 32, 64, 128, 256]
        _output_channels_list = [8, 16, 32, 64, 128, 256, 512]

        self.conv_layers.append(
            SqueezeExcitation(input_channels=input_channels, reduced_dim=3)
        )

        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    _input_channels_list[i], _output_channels_list[i], kernel_size=3, padding=1,
                    bias=True
                )
            )
            if _output_channels_list[i] == 64:
                self.conv_layers.append(
                    SqueezeExcitation(
                        input_channels=_output_channels_list[i],
                        reduced_dim=int(_output_channels_list[i] / 4)
                    )
                )
            self.conv_layers.append(nn.BatchNorm2d(num_features=_output_channels_list[i]))
            self.conv_layers.append(nn.ReLU())

        self.drop_out = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Conv2d(512, output_channel, kernel_size=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.drop_out(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)


class MCDCNN_New(nn.Module):
    def __init__(self, dropout=0.3, input_channels=4, output_channel=1, num_layers=6):
        super(MCDCNN_New, self).__init__()

        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()

        _input_channels_list = [input_channels, 8, 16, 32, 64, 128]
        _output_channels_list = [8, 16, 32, 64, 128, 256]

        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    _input_channels_list[i], _output_channels_list[i], kernel_size=3, padding=1,
                    bias=True
                )
            )
            # self.conv_layers.append(nn.BatchNorm2d(num_features=_output_channels_list[i]))
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.Dropout2d(p=dropout))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.final_conv = nn.Conv2d(1024, output_channel, kernel_size=1)

    def forward(self, img1, img2, img3, img4):
        ch1_out = img1
        for layer in self.conv_layers:
            ch1_out = layer(ch1_out)

        ch2_out = img2
        for layer in self.conv_layers:
            ch2_out = layer(ch2_out)

        ch3_out = img3
        for layer in self.conv_layers:
            ch3_out = layer(ch3_out)

        ch4_out = img4
        for layer in self.conv_layers:
            ch4_out = layer(ch4_out)

        x = torch.cat([ch1_out, ch2_out, ch3_out, ch4_out], dim=1)
        # x = self.max_pool(x)
        # print('eeee', x.shape)
        x = self.final_conv(x)
        return torch.sigmoid(x)
