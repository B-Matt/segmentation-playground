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
    def __init__(self, dropout = 0.3, resolution = 256, input_channels = 4, output_channel = 1, num_layers = 8):
        super(FFMMNet, self).__init__()
	
        self.conv_layers = nn.ModuleList()
        _input_channels_list = [input_channels, 32, 48, 64, 80, 96, 112, 128]
        _output_channels_list = [32, 48, 64, 80, 96, 112, 128, 144]

        for i in range(num_layers):
            self.conv_layers.append(nn.Conv2d(_input_channels_list[i], _output_channels_list[i], kernel_size=3, padding=1, bias=False))
            self.conv_layers.append(nn.BatchNorm2d(num_features=_output_channels_list[i]))
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.Dropout(p=dropout))

        self.start_se = SqueezeExcitation(input_channels, reduced_dim=2)

        self.max_pool = nn.AdaptiveMaxPool2d((round(resolution // 2), round(resolution // 2)))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.middle_se1 = SqueezeExcitation(80, reduced_dim=28)

        self.end_se = SqueezeExcitation(224, reduced_dim=86)
        self.drop_out = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Sequential(
            nn.Conv2d(224, output_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1)
        )
    
    def forward(self, x):
        # Branch #1
        branch_1_x = self.start_se(x)
        for i in range(0, len(self.conv_layers) // 2):
            branch_1_x = self.conv_layers[i](branch_1_x)

        # Branch #2
        branch_2_x = self.max_pool(branch_1_x)
        for i in range(len(self.conv_layers) // 2, len(self.conv_layers)):
            branch_2_x = self.conv_layers[i](branch_2_x)

        # Up Scaling
        x_upscaled_1 = self.up(branch_2_x)
        x_upscaled_1 = torch.cat((x_upscaled_1, self.middle_se1(branch_1_x)), dim=1)

        # Finish
        x = self.end_se(x_upscaled_1)
        # x = self.drop_out(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)
