import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDCNN(nn.Module):
    def __init__(self, num_layers=6):
        super(MCDCNN, self).__init__()

        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
	
        # Jer je input broj layera 5, 1-64, 64-32, 32-16, 16-8, 8-1
        _input_channels_list = [5, 128, 64, 32, 16, 8] #[5, 64, 32, 16, 8]
        _output_channels_list = [128, 64, 32, 16, 8, 1] #[64, 32, 16, 8, 1]	

        for i in range(num_layers):
            self.conv_layers.append(nn.Conv2d(_input_channels_list[i], _output_channels_list[i], kernel_size=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout2d(p=0.1))
        
        self.final_conv = nn.Conv2d(1, 1, kernel_size=1)
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        x = self.final_conv(x)
        # x = F.softmax(x, dim=1)
        # x = nn.Softmax(x)
        return x