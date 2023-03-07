import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MCDCNN(nn.Module):
    def __init__(self, dropout = 0.3, input_channels = 4, output_channel = 1):
        super(MCDCNN, self).__init__()
	
        # _input_channels_list = [input_channels, 32, 64, 128, 256, 512]
        # _output_channels_list = [32, 64, 128, 256, 512, 1024] 

        # _input_channels_list = [input_channels, 32, 64, 128, 256, 512]
        # _output_channels_list = [32, 64, 128, 256, 512, 1024]

        _input_channels_list = [input_channels, 16, 32, 64, 128, 256, 512]
        _output_channels_list = [16, 32, 64, 128, 256, 512, 1024]

        self.conv1 = ConvLayer(_input_channels_list[0], _output_channels_list[0])
        self.conv2 = ConvLayer(_input_channels_list[1], _output_channels_list[1])
        self.conv3 = ConvLayer(_input_channels_list[2], _output_channels_list[2])
        self.conv4 = ConvLayer(_input_channels_list[3], _output_channels_list[3])
        self.conv5 = ConvLayer(_input_channels_list[4], _output_channels_list[4])
        self.conv6 = ConvLayer(_input_channels_list[5], _output_channels_list[5])
        self.conv7 = ConvLayer(_input_channels_list[6], _output_channels_list[6])

        self.final_conv = nn.Conv2d(1024, output_channel, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.final_conv(x)
        x = self.dropout(x)
        return torch.sigmoid(x)