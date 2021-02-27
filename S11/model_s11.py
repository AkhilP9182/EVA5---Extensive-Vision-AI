import torch
import torch.nn as nn
import torch.nn.functional as F

class model_s11(nn.Module):
    def __init__(self, num_classes=10):
        super(model_s11, self).__init__()
        #---------------- Prep Layer -------------#
        self.prepblock  = nn.Sequential(
                                                                   #     INPUT     |    OUTPUT      | Receptive Field

            nn.Conv2d(3, 64, 3, stride=1, padding=1,bias=False),   # In: 32x32x3   | Out: 32x32x64  |      RF:3
            nn.BatchNorm2d(64),                                    # In: 32x32x64  | Out: 32x32x64  |      RF:3
            nn.ReLU()                                              # In: 32x32x64  | Out: 32x32x64  |      RF:3
        )

        #---------------- Layer 1 ----------------#
        self.convblock_L1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1,bias=False), # In: 32x32x64  | Out: 32x32x128 |      RF:5
            nn.MaxPool2d(2, 2),                                    # In: 32x32x128 | Out: 16x16x128 |      RF:6
            nn.BatchNorm2d(128),                                   # In: 16x16x128 | Out: 16x16x128 |      RF:6
            nn.ReLU()                                              # In: 16x16x128 | Out: 16x16x128 |      RF:6
        )

        self.Resblock_L1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1,bias=False),# In: 16x16x128 | Out: 16x16x128 |      RF:10
            nn.BatchNorm2d(128),                                   # In: 16x16x128 | Out: 16x16x128 |      RF:10
            nn.ReLU(),                                             # In: 16x16x128 | Out: 16x16x128 |      RF:10

            nn.Conv2d(128, 128, 3, stride=1, padding=1,bias=False),# In: 16x16x128 | Out: 16x16x128 |      RF:14
            nn.BatchNorm2d(128),                                   # In: 16x16x128 | Out: 16x16x128 |      RF:14
            nn.ReLU()                                              # In: 16x16x128 | Out: 16x16x128 |      RF:14
        )

        #---------------- Layer 2 ----------------#
        self.convblock_L2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1,bias=False),# In: 16x16x128 | Out: 16x16x256 |      RF:18
            nn.MaxPool2d(2, 2),                                    # In: 16x16x256 | Out: 8x8x256   |      RF:20
            nn.BatchNorm2d(256),                                   # In: 8x8x256   | Out: 8x8x256   |      RF:20
            nn.ReLU()                                              # In: 8x8x256   | Out: 8x8x256   |      RF:20
        )

        #---------------- Layer 3 ----------------#
        self.convblock_L3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1,bias=False),# In: 8x8x256   | Out: 8x8x512   |      RF:28
            nn.MaxPool2d(2, 2),                                    # In: 8x8x512   | Out: 4x4x512   |      RF:28
            nn.BatchNorm2d(512),                                   # In: 4x4x512   | Out: 4x4x512   |      RF:28
            nn.ReLU()                                              # In: 4x4x512   | Out: 4x4x512   |      RF:28
        )

        self.Resblock_L3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1,bias=False),# In: 4x4x512   | Out: 4x4x512   |      RF:36
            nn.BatchNorm2d(512),                                   # In: 4x4x512   | Out: 4x4x512   |      RF:36
            nn.ReLU(),                                             # In: 4x4x512   | Out: 4x4x512   |      RF:36

            nn.Conv2d(512, 512, 3, stride=1, padding=1,bias=False),# In: 4x4x512   | Out: 4x4x512   |      RF:44
            nn.BatchNorm2d(512),                                   # In: 4x4x512   | Out: 4x4x512   |      RF:44
            nn.ReLU()                                              # In: 4x4x512   | Out: 4x4x512   |      RF:44
        )

        #---------------- Layer 4 ----------------#
        self.maxpool_L4 = nn.MaxPool2d(kernel_size=4, stride=1)    # In: 4x4x512   | Out: 1x1x512   |      RF:47
        self.linear_L4  = nn.Linear(512, num_classes)              # In: 512       | Out: 1 x 10

    def forward(self, x):
        '''
            Method for passing the input image through the network to get the output
            x here is a tensor at each stage, with the dimensions [N, C, H, W]:- 
            {N = No. of Samples, C = No. of channels, H = Height of Image, W = Width of Image}
        '''
        #---------------- Prep Layer -------------#
        x = self.prepblock(x)

        #---------------- Layer 1 ----------------#
        x = self.convblock_L1(x)
        Res1 = x.clone()
        x = self.Resblock_L1(x)
        x = x + Res1

        #---------------- Layer 2 ----------------#
        x = self.convblock_L2(x)

        #---------------- Layer 3 ----------------#
        x = self.convblock_L3(x)
        Res2 = x.clone()
        x = self.Resblock_L3(x)
        x = x + Res2

        #---------------- Layer 4 ----------------#
        x = self.maxpool_L4(x)
        x = torch.flatten(x, 1)
        x = self.linear_L4(x)

        return F.log_softmax(x,dim=1)