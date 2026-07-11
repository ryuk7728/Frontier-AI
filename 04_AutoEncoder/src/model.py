import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            ResidualBlock(16),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            ResidualBlock(32),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(64*4*4,256)
        )

        self.decoder = nn.Sequential(

            nn.Linear(256,1024),

            nn.Unflatten(1,(64,4,4)), # Essentially does a reshape 

            nn.ConvTranspose2d( # Converts to 32*8*8 by multiplying each input element by the kernel and shifting by 2 steps to essentially double 4*4 to 8*8
                64,32,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,16,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(
                16,3,
                kernel_size=2,
                stride=2
            ),

            nn.Sigmoid()
            )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
