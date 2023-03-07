import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDCNN(nn.Module):
    def __init__(self, dropout = 0.3, input_channels = 4, output_channel = 1, num_layers = 6):
        super(MCDCNN, self).__init__()
	
        self.conv_layers = nn.ModuleList()
        _input_channels_list = [input_channels, 8, 16, 32, 64, 128]
        _output_channels_list = [8, 16, 32, 64, 128, 256]

        for i in range(num_layers):
            self.conv_layers.append(nn.Conv2d(_input_channels_list[i], _output_channels_list[i], kernel_size=3, padding=1, bias=True))
            self.conv_layers.append(nn.BatchNorm2d(num_features=_output_channels_list[i]))
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.Dropout2d(p=dropout))

        self.final_conv = nn.Conv2d(256, output_channel, kernel_size=1)
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.final_conv(x)
        return torch.sigmoid(x)

class MCDCNN_New(nn.Module):
    def __init__(self, dropout = 0.3, input_channels = 4, output_channel = 1, num_layers = 6):
        super(MCDCNN_New, self).__init__()
	
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
	
        _input_channels_list = [input_channels, 8, 16, 32, 64, 128]
        _output_channels_list = [8, 16, 32, 64, 128, 256]

        for i in range(num_layers):
            self.conv_layers.append(nn.Conv2d(_input_channels_list[i], _output_channels_list[i], kernel_size=3, padding=1, bias=True))
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
