import torch
from torch import nn

DIM = 64 # This overfits substantially; you're probably better off with 64

class Generator(nn.Module):
    def __init__(self, input_channel = 512, output_channel = 512):
        super(Generator, self).__init__()
                
        preprocess = nn.Sequential(
            nn.Conv2d(input_channel, DIM, kernel_size=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(DIM, 2 * DIM, kernel_size =2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, kernel_size =1),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        
        deconv_out = nn.ConvTranspose2d(DIM, output_channel, kernel_size =1) 

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        input = self.preprocess(input)
        output = self.block1(input)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_channel = 512):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(input_channel, DIM, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.main = main
        self.linear = nn.Linear(4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*DIM)
        output = self.linear(output)
        return output