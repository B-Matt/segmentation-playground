import torch
import torch.nn as nn

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

class FFMMNet2(nn.Module):
    def __init__(self, dropout = 0.1, resolution = 256, input_channels = 4, output_channel = 1):
        super(FFMMNet2, self).__init__()
        _input_channels_list = [input_channels, 16, 32, 48, 64, 80, 96]
        _output_channels_list = [16, 32, 48, 64, 80, 96, 122]

        self.down_branch_layers_1 = nn.Sequential(
            nn.Conv2d(_input_channels_list[0], _output_channels_list[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(_input_channels_list[1], _output_channels_list[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[1]),
            nn.ReLU(inplace=True),
        )

        self.down_branch_layers_2 = nn.Sequential(
            nn.Conv2d(_input_channels_list[2], _output_channels_list[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(_input_channels_list[3], _output_channels_list[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[3]),
            nn.ReLU(inplace=True),
        )

        self.down_branch_layers_3 = nn.Sequential(
            nn.Conv2d(_input_channels_list[4], _output_channels_list[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[4]),
            nn.ReLU(inplace=True),

            nn.Conv2d(_input_channels_list[5], _output_channels_list[5], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[5]),
            nn.ReLU(inplace=True),

            nn.Conv2d(_input_channels_list[6], _output_channels_list[6], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[6]),
            nn.ReLU(inplace=True),
        )

        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.max_pool1 = nn.AdaptiveMaxPool2d((round(resolution // 2), round(resolution // 2)))
        self.max_pool2 = nn.AdaptiveMaxPool2d((round(resolution // 4), round(resolution // 4)))

        self.end_se = SqueezeExcitation(218, reduced_dim=115)
        self.down_linear = nn.Sequential(
            nn.Conv2d(218, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True)
        )

        self.end_conv_layer = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1,  bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1,  bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
        )

        self.drop_out = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, output_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=output_channel)
        )
    
    def forward(self, masks, mean_std):
        # Upper Branch
        up_branch = mean_std

        # Lower Branch
        down_branch_1 = self.down_branch_layers_1(masks)
        down_branch_2 = self.max_pool1(down_branch_1)
        down_branch_2 = self.down_branch_layers_2(down_branch_2)

        down_branch_3 = self.max_pool2(down_branch_2)
        down_branch_3 = self.down_branch_layers_3(down_branch_3)

        up_sampled_2 = self.up(down_branch_3)
        up_sampled_2 = torch.cat((up_sampled_2, down_branch_2), dim=1)

        up_sampled_1 = self.up(up_sampled_2)
        down_branch_final = torch.cat((up_sampled_1, down_branch_1), dim=1)

        down_branch_final = self.end_se(down_branch_final)
        down_branch_final = self.down_linear(down_branch_final)

        # Middle Branch
        middle_branch = torch.cat((up_branch, down_branch_final), dim=1)

        # Ending
        x = self.end_conv_layer(middle_branch)
        x = self.drop_out(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)

class FFMMNet1(nn.Module):
    def __init__(self, dropout = 0.1, resolution = 256, input_channels = 4, output_channel = 1):
        super(FFMMNet1, self).__init__()
        _input_channels_list = [input_channels, 16, 32, 48, 64, 80]
        _output_channels_list = [16, 32, 48, 64, 80, 96]

        self.down_branch_layers_1 = nn.Sequential(
            nn.Conv2d(_input_channels_list[0], _output_channels_list[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(_input_channels_list[1], _output_channels_list[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[1]),
            nn.ReLU(inplace=True),
        )

        self.down_branch_layers_2 = nn.Sequential(
            nn.Conv2d(_input_channels_list[2], _output_channels_list[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(_input_channels_list[3], _output_channels_list[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[3]),
            nn.ReLU(inplace=True),
        )

        self.down_branch_layers_3 = nn.Sequential(
            nn.Conv2d(_input_channels_list[4], _output_channels_list[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[4]),
            nn.ReLU(inplace=True),

            nn.Conv2d(_input_channels_list[5], _output_channels_list[5], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_output_channels_list[5]),
            nn.ReLU(inplace=True),
        )

        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.max_pool1 = nn.AdaptiveMaxPool2d((round(resolution // 2), round(resolution // 2)))
        self.max_pool2 = nn.AdaptiveMaxPool2d((round(resolution // 4), round(resolution // 4)))

        self.end_se = SqueezeExcitation(192, reduced_dim=115)
        self.down_linear = nn.Sequential(
            nn.Conv2d(192, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True)
        )

        self.end_conv_layer = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1,  bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1,  bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
        )

        self.drop_out = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, output_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=output_channel)
        )
    
    def forward(self, masks, mean_std):
        # Upper Branch
        up_branch = mean_std

        # Lower Branch
        down_branch_1 = self.down_branch_layers_1(masks)
        down_branch_2 = self.max_pool1(down_branch_1)
        down_branch_2 = self.down_branch_layers_2(down_branch_2)

        down_branch_3 = self.max_pool2(down_branch_2)
        down_branch_3 = self.down_branch_layers_3(down_branch_3)

        up_sampled_2 = self.up(down_branch_3)
        up_sampled_2 = torch.cat((up_sampled_2, down_branch_2), dim=1)

        up_sampled_1 = self.up(up_sampled_2)
        down_branch_final = torch.cat((up_sampled_1, down_branch_1), dim=1)

        down_branch_final = self.end_se(down_branch_final)
        down_branch_final = self.down_linear(down_branch_final)

        # Middle Branch
        middle_branch = torch.cat((up_branch, down_branch_final), dim=1)

        # Ending
        x = self.end_conv_layer(middle_branch)
        x = self.drop_out(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)