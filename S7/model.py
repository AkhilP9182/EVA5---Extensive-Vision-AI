import torch
import torch.nn as nn
import torch.nn.functional as F
from S7.layers import BatchNorm,GhostBatchNorm,Depth_Sep_Conv

class Net(nn.Module):
    '''
        nn.Module is the base class for all Neural Network Modules, the Net() class is inheriting the base class nn.Module
        A Module contains the state of the layers in a Neural network and methods for feedforward and training the model
    '''
    # BN_flag 0: normal batchnorm; 1: Ghost batchnorm
    
    

    def __init__(self,BN_type='BN'):
        def BN_Layer(self,channels,BN_type='BN'):
            '''
                BN_type == 'BN' -> GhostBatchNorm()
                BN_type == 'GBN' -> nn.BatchNorm2d()
                Selects the type of Batch Normalization which is to be used
            '''
            if BN_type == 'GBN':
                return GhostBatchNorm(channels, num_splits=2, weight=False)
            elif BN_type == 'BN':
                return nn.BatchNorm2d(channels)

        super(Net, self).__init__() 
        # Conv Block 1
        drop = 0.05          # Dropout Percentage
        self.convblock1 = nn.Sequential(
                                                                   #     INPUT     |    OUTPUT      | Receptive Field

            nn.Conv2d(3, 32, 3,padding=1,bias=False),              # In: 32x32x3   | Out: 32x32x32  |      RF:3
            BN_Layer(self,32,BN_type),                             # In: 32x32x32  | Out: 32x32x32  |      RF:3
            nn.ReLU(),                                             # In: 32x32x32  | Out: 32x32x32  |      RF:3
            nn.Dropout(p=drop),                                    # In: 32x32x32  | Out: 32x32x32  |      RF:3

            nn.Conv2d(32, 64, 3,padding=1,bias=False),             # In: 32x32x32  | Out: 32x32x64  |      RF:5
            BN_Layer(self,64,BN_type),                             # In: 32x32x64  | Out: 32x32x64  |      RF:5
            nn.ReLU(),                                             # In: 32x32x64  | Out: 32x32x64  |      RF:5
            nn.Dropout(p=drop),                                    # In: 32x32x64  | Out: 32x32x64  |      RF:5

            nn.Conv2d(64, 128, 3,bias=False,padding=0,dilation=2), # In: 32x32x64  | Out: 28x28x128 |      RF:9    ---> Dilated Convolution
            BN_Layer(self,128,BN_type),                            # In: 28x28x32  | Out: 28x28x32  |      RF:9
            nn.ReLU(),                                             # In: 28x28x32  | Out: 28x28x32  |      RF:9
            nn.Dropout(p=drop)                                     # In: 28x28x32  | Out: 28x28x32  |      RF:9
        )
        
        # Transition 1 
        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                    # In: 28x28x32  | Out: 14x14x32  |      RF:10
            nn.Conv2d(128, 32, 1,bias=False),                      # In: 14x14x128 | Out: 14x14x32  |      RF:10    ---> Pointwise Convolution
            BN_Layer(self,32,BN_type),                             # In: 14x14x32  | Out: 14x14x32  |      RF:10
            nn.ReLU(),                                             # In: 14x14x32  | Out: 14x14x32  |      RF:10
            nn.Dropout(p=drop)                                     # In: 14x14x32  | Out: 14x14x32  |      RF:10
        )

        # Conv Block 2 
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 64, 3,bias=False,padding=1),             # In: 14x14x32  | Out: 14x14x64  |      RF:14
            BN_Layer(self,64,BN_type),                             # In: 14x14x64  | Out: 14x14x64  |      RF:14
            nn.ReLU(),                                             # In: 14x14x64  | Out: 14x14x64  |      RF:14
            nn.Dropout(p=drop),                                    # In: 14x14x64  | Out: 14x14x64  |      RF:14

            nn.Conv2d(64, 128, 3,bias=False,padding=1),            # In: 14x14x32  | Out: 14x14x64  |      RF:18
            BN_Layer(self,128,BN_type),                            # In: 14x14x64  | Out: 14x14x64  |      RF:18
            nn.ReLU(),                                             # In: 14x14x64  | Out: 14x14x64  |      RF:18
            nn.Dropout(p=drop),                                    # In: 14x14x64  | Out: 14x14x64  |      RF:18

            Depth_Sep_Conv(128, 256, 3,padding=1),                 # In: 14x14x64  | Out: 14x14x256 |      RF:22    ---> Depthwise Separable Convolution
            BN_Layer(self,256,BN_type),                            # In: 14x14x256 | Out: 14x14x256 |      RF:22
            nn.ReLU(),                                             # In: 14x14x256 | Out: 14x14x256 |      RF:22
            nn.Dropout(p=drop)                                     # In: 14x14x256 | Out: 14x14x256 |      RF:22
        )

        # Transition 1 
        self.transblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                    # In: 14x14x256 | Out: 7x7x256   |      RF:24
            nn.Conv2d(256, 64, 1,bias=False),                      # In: 7x7x256   | Out: 7x7x64    |      RF:24    ---> Pointwise Convolution
            BN_Layer(self,64,BN_type),                             # In: 7x7x64    | Out: 7x7x64    |      RF:24
            nn.ReLU(),                                             # In: 7x7x64    | Out: 7x7x64    |      RF:24
            nn.Dropout(p=drop)                                     # In: 7x7x64    | Out: 7x7x64    |      RF:24
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(64, 128, 3,bias=False,padding=1),            # In: 7x7x32    | Out: 7x7x32    |      RF:32
            BN_Layer(self,128,BN_type),                            # In: 7x7x32    | Out: 7x7x32    |      RF:32
            nn.ReLU(),                                             # In: 7x7x32    | Out: 7x7x32    |      RF:32
            nn.Dropout(p=drop),                                    # In: 7x7x32    | Out: 7x7x32    |      RF:32

            nn.Conv2d(128, 128, 3,bias=False,padding=1),           # In: 7x7x32    | Out: 7x7x64    |      RF:40
            BN_Layer(self,128,BN_type),                            # In: 7x7x64    | Out: 7x7x64    |      RF:40
            nn.ReLU(),                                             # In: 7x7x64    | Out: 7x7x64    |      RF:40
            nn.Dropout(p=drop),                                    # In: 7x7x64    | Out: 7x7x64    |      RF:40
  
            Depth_Sep_Conv(128, 256, 3,padding=1),                 # In: 7x7x64    | Out: 7x7x128   |      RF:48    ---> Depthwise Separable Convolution
            BN_Layer(self,256,BN_type),                            # In: 7x7x128   | Out: 7x7x128   |      RF:48
            nn.ReLU(),                                             # In: 7x7x128   | Out: 7x7x128   |      RF:48
            nn.Dropout(p=drop)                                     # In: 7x7x128   | Out: 7x7x128   |      RF:48
        )
        # Output Block                
        self.outblock = nn.Sequential(
            nn.Conv2d(256, 64, 3,bias=False),                      # In: 7x7x256   | Out:  5x5x64    |     RF:56
            nn.Conv2d(64, num_classes, 3,bias=False),              # In: 5x5x256   | Out:  3x3x64    |     RF:64
            nn.AvgPool2d(kernel_size=3)                            # In: 3x3x10    | Out:  1x1x10    |     RF:64
        )
        

    def forward(self, x):
        '''
            Method for passing the input image through the network to get the output
            x here is a tensor at each stage, with the dimensions [N, C, H, W]:- 
            {N = No. of Samples, C = No. of channels, H = Height of Image, W = Width of Image}
        '''
        x = self.convblock1(x)
        x = self.transblock1(x)
        x = self.convblock2(x)
        x = self.transblock2(x)
        x = self.convblock3(x)
        x = self.outblock(x)
        x = x.view(-1, 10)               # Reshaping the tensor to a tensor with 10 columns and appropriate numbe of rows
        return F.log_softmax(x,dim=1)    # Computing Softmax of the obtained output
