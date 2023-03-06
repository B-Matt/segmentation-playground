import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDCNN(nn.Module):
    def __init__(self, dropout = 0.1, input_channels = 5, output_channel = 1, num_layers = 6):
        super(MCDCNN, self).__init__()

        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
	
        _input_channels_list = [input_channels, 32, 64, 128, 256, 512]
        _output_channels_list = [32, 64, 128, 256, 512, 1024] 

        for i in range(num_layers):
            self.conv_layers.append(nn.Conv2d(_input_channels_list[i], _output_channels_list[i], kernel_size=1, stride=1, padding=0))
            self.conv_layers.append(nn.BatchNorm2d(num_features=_output_channels_list[i]))
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.Dropout2d(p=dropout))
        
        self.final_conv = nn.Conv2d(1024, output_channel, kernel_size=1)
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.final_conv(x)
        return torch.sigmoid(x)