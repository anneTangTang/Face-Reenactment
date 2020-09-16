import torch
import torch.nn as nn

from utils import bilinear_interpolation


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        # this layer is Conv or deConv
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm2d") != -1:
        # this layer is BatchNorm2d
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


########################
#        U-NET
########################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (n, h, w, in_size)
        Returns:
            self.model(x): (n, h/2, w/2, out_size)
        """
        return self.model(x)


class SimpleUp(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=True, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_size))
        if dropout != 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))

        self.layers = layers


class Refine(nn.Module):
    def __init__(self, in_size, out_size, is_last=False, batch_norm=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_size))
        if dropout != 0.0:
            layers.append(nn.Dropout(dropout))
        if not is_last:
            layers.append(nn.ReLU(inplace=True))

        self.layers = layers


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, is_last=False, batch_norm=True, dropout=0.0):
        super(UNetUp, self).__init__()
        simpleUp = SimpleUp(in_size, out_size, batch_norm, dropout)
        refine1 = Refine(out_size, out_size, is_last, batch_norm, dropout)
        refine2 = Refine(out_size, out_size, is_last, batch_norm, dropout)

        layers = simpleUp.layers + refine1.layers + refine2.layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input=None, downsample_input=None):
        """
        Args:
            x: (n, channel_1, h/2, w/2)
            skip_input: (n, channel_2, h, w)
            downsample_input: (n, 9Nw, h, w). the bi-linear downsampling of input X
        Returns:
            (n, out_size, h, w)
        """
        x = self.model(x)  # (n, channel, h, w)
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # (n, channel + channel_2, h, w)
        if downsample_input is not None:
            x = torch.cat((x, downsample_input), 1)  # (n, channel + channel_2 + 9Nw, h, w)

        return x


########################
#        RESNET
########################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


########################
#      Generator
########################


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=99, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, batch_norm=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 512, dropout=0.5)
        # self.up4 = UNetUp(1024, 512)
        # self.up5 = UNetUp(1024, 256)
        self.up2 = UNetUp(512, 512, dropout=0.5)
        self.up3 = UNetUp(512, 512, dropout=0.5)
        self.up4 = UNetUp(512, 512)
        self.up5 = UNetUp(512, 256)
        self.up6 = UNetUp(512 + in_channels, 128)
        self.up7 = UNetUp(256 + in_channels, 64)
        self.up8 = UNetUp(128 + in_channels, out_channels, batch_norm=False)

        self.final = nn.Sequential(nn.Tanh(),)

    def forward(self, x):
        """U-Net generator with skip connections from encoder to decoder.

        :param x: (n, 9Nw, h, w). numpy
        :return: (n, 3, h, w).
        """

        """
        Question 1: If the data format of input x can be ndarray.
        Answer 1: During loading dataset, we have already transformed the ndarray to tensor. 
        Question 2: If the parameter of function 'torch.cat' can be ndarray.
        Answer2: Because the input of G and D has been transformed to torch.Tensor, 
        these intermediate generated variables are tensor too.
        """

        d1 = self.down1(x)  # (n, 64, h/2, w/2)
        d2 = self.down2(d1)  # (n, 128, h/4, w/4)
        d3 = self.down3(d2)  # (n, 256, h/8, w/8)
        d4 = self.down4(d3)  # (n, 512, h/16, w/16)
        d5 = self.down5(d4)  # (n, 512, h/32, w/32)
        d6 = self.down6(d5)  # (n, 512, h/64, w/64)
        d7 = self.down7(d6)  # (n, 512, h/128, w/128)
        d8 = self.down8(d7)  # (n, 512, h/256, w/256)

        # Calculate the bi-linear down-sampling of input x.
        downsample_32 = bilinear_interpolation(x, 32, 32)  # (n, 9Nw, 32, 32). torch.cuda.FloatTensor
        downsample_64 = bilinear_interpolation(x, 64, 64)  # (n, 9Nw, 64, 64). torch.cuda.FloatTensor
        downsample_128 = bilinear_interpolation(x, 128, 128)  # (n, 9Nw, 128, 128). torch.cuda.FloatTensor

        u1 = self.up1(d8, skip_input=d7)  # (n, 1024, h/128, w/128)
        u2 = self.up2(u1, skip_input=d6)  # (n, 1024, h/64, w/64)
        u3 = self.up3(u2, skip_input=d5)  # (n, 1024, h/32, w/32)
        u4 = self.up4(u3, skip_input=d4)  # (n, 1024, h/16, w/16)
        u5 = self.up5(u4, skip_input=d3, downsample_input=downsample_32)  # (n, 256+256+9Nw, h/8, w/8)
        u6 = self.up6(u5, skip_input=d2, downsample_input=downsample_64)  # (n, 128+128+9Nw, h/4, w/4)
        u7 = self.up7(u6, skip_input=d1, downsample_input=downsample_128)  # (n, 64+64+9Nw, h/2, w/2)
        u8 = self.up8(u7)  # (n, 3, h, w)

        return self.final(u8)


########################
#     Discriminator
########################
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        def discriminator_block2d(in_filters, out_filters, is_first, max_pooling=True):
            """Returns downsampling layers of each discriminator block."""
            if is_first:
                layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            else:
                layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]

            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if max_pooling:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels=out_filters, out_channels=1, kernel_size=1, stride=1))
            return layers

        self.model = nn.Sequential(
            *discriminator_block2d(in_channels, 32, is_first=True),
            *discriminator_block2d(32, 64, is_first=False),
            *discriminator_block2d(64, 128, is_first=False),
            *discriminator_block2d(128, 256, is_first=False, max_pooling=False),
            nn.Sigmoid()
        )

    def forward(self, G_output, G_input):
        """Concatenate generated image and condition image by channels to produce input.

        :param G_output: (n, 3, h, w)
        :param G_input: (n, 9Nw, h, w)
        :return: (n, 1, h/8, w/8)
        """
        D_input = torch.cat((G_output, G_input), 1)  # (n, 3+9Nw, h, w)
        return self.model(D_input)


########################
#   Residual Generator
########################


class GeneratorRes64(nn.Module):
    def __init__(self, in_channels=6, out_channels=4, skip=False):
        super().__init__()
        self._skip = skip

        self.down1 = UNetDown(in_channels, 64, batch_norm=False)
        self.down2 = UNetDown(64, 128)

        self.resblock1 = ResidualBlock(128)
        self.resblock2 = ResidualBlock(128)
        self.resblock3 = ResidualBlock(128)
        self.resblock4 = ResidualBlock(128)
        self.resblock5 = ResidualBlock(128)
        self.resblock6 = ResidualBlock(128)
        self.resblock7 = ResidualBlock(128)
        self.resblock8 = ResidualBlock(128)
        self.resblock9 = ResidualBlock(128)

        if not self._skip:
            self.up1 = UNetUp(128 + in_channels, 64, is_last=False)
            self.up2 = UNetUp(64 + in_channels, out_channels, is_last=True, batch_norm=False)
        else:
            self.up1 = UNetUp(128 + 128 + in_channels, 64, is_last=False)
            self.up2 = UNetUp(64 + 64 + in_channels, out_channels, is_last=True, batch_norm=False)

        self.final = nn.Sequential(nn.Tanh())

    def forward(self, x):
        """U-Net generator with skip connections from encoder to decoder.

        :param x: (n, 9Nw, h, w).
        :return: (n, 3, h, w).
        """

        """
        Question 1: If the data format of input x can be ndarray.
        Answer 1: During loading dataset, we have already transformed the ndarray to torch tensor. 
        Question 2: If the parameter of function "torch.cat" can be ndarray.
        Answer2: Because the input of G and D has been transformed to torch.Tensor, these intermediate generated variables are tensor too.
        """

        d1 = self.down1(x)  # (n, 64, 128, 128)
        d2 = self.down2(d1)  # (n, 128, 64, 64)

        res1 = self.resblock1(d2)
        res2 = self.resblock2(res1)
        res3 = self.resblock3(res2)
        res4 = self.resblock4(res3)
        res5 = self.resblock5(res4)
        res6 = self.resblock6(res5)
        res7 = self.resblock7(res6)
        res8 = self.resblock8(res7)
        res9 = self.resblock9(res8)  # (n, 128, 64, 64)

        # Calculate the bi-linear downsampling of input x.
        downsample_64 = bilinear_interpolation(x, 64, 64)  # (n, 9Nw, 64, 64). FloatTensor
        downsample_128 = bilinear_interpolation(x, 128, 128)  # (n, 9Nw, 128, 128). FloatTensor

        if not self._skip:
            b = torch.cat((res9, downsample_64), 1)  # (n, 128+9Nw, 64, 64)
            u1 = self.up1(b, downsample_input=downsample_128)  # (n, 64+9Nw, 128, 128)
        else:
            b = torch.cat((res9, d2, downsample_64), 1)  # (n, 128+128+9Nw, 64, 64)
            u1 = self.up1(b, skip_input=d1, downsample_input=downsample_128)  # (n, 64+64+9Nw, 128, 128)

        u2 = self.up2(u1)  # (n, 3, 256, 256)

        return self.final(u2)


class GeneratorRes32(nn.Module):
    def __init__(self, in_channels=99, out_channels=3, skip=False):
        super().__init__()

        self._skip = skip

        self.down1 = UNetDown(in_channels, 64, batch_norm=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)

        self.resblock1 = ResidualBlock(256)
        self.resblock2 = ResidualBlock(256)
        self.resblock3 = ResidualBlock(256)
        self.resblock4 = ResidualBlock(256)
        self.resblock5 = ResidualBlock(256)
        self.resblock6 = ResidualBlock(256)
        self.resblock7 = ResidualBlock(256)
        self.resblock8 = ResidualBlock(256)
        self.resblock9 = ResidualBlock(256)

        if not self._skip:
            self.up1 = UNetUp(256 + in_channels, 128, is_last=False)
            self.up2 = UNetUp(128 + in_channels, 64, is_last=False)
            self.up3 = UNetUp(64 + in_channels, out_channels, is_last=True, batch_norm=False)
        else:
            self.up1 = UNetUp(256 + 256 + in_channels, 128, is_last=False)
            self.up2 = UNetUp(128 + 128 + in_channels, 64, is_last=False)
            self.up3 = UNetUp(64 + 64 + in_channels, out_channels, is_last=True, batch_norm=False)

        self.final = nn.Sequential(nn.Tanh(),)

    def forward(self, x):
        """U-Net generator with skip connections from encoder to decoder.

        :param x: (n, 9Nw, h, w).
        :return: (n, 3, h, w).
        """

        d1 = self.down1(x)  # (n, 64, 128, 128)
        d2 = self.down2(d1)  # (n, 128, 64, 64)
        d3 = self.down3(d2)  # (n, 256, 32, 32)

        res1 = self.resblock1(d3)
        res2 = self.resblock2(res1)
        res3 = self.resblock3(res2)
        res4 = self.resblock4(res3)
        res5 = self.resblock5(res4)
        res6 = self.resblock6(res5)
        res7 = self.resblock7(res6)
        res8 = self.resblock8(res7)
        res9 = self.resblock9(res8)  # (n, 256, 32, 32)

        # Calculate the bilinear downsampling of input x.
        downsample_32 = bilinear_interpolation(x, 32, 32)  # (n, 9Nw, 32, 32). FloatTensor
        downsample_64 = bilinear_interpolation(x, 64, 64)  # (n, 9Nw, 64, 64). FloatTensor
        downsample_128 = bilinear_interpolation(x, 128, 128)  # (n, 9Nw, 128, 128). FloatTensor

        if not self._skip:
            b = torch.cat((res9, downsample_32), 1)  # (n, 256+256+9Nw, 32, 32)
            u1 = self.up1(b, downsample_input=downsample_64)  # (n, 128+128+9Nw, 64, 64)
            u2 = self.up2(u1, downsample_input=downsample_128)  # (n, 64+64+9Nw, 128, 128)
        else:
            b = torch.cat((res9, d3, downsample_32), 1)  # (n, 256+256+9Nw, 32, 32)
            u1 = self.up1(b, skip_input=d2, downsample_input=downsample_64)  # (n, 128+128+9Nw, 64, 64)
            u2 = self.up2(u1, skip_input=d1, downsample_input=downsample_128)  # (n, 64+64+9Nw, 128, 128)

        u3 = self.up3(u2)  # (n, 3, 256, 256)

        return self.final(u3)


class GeneratorRes16(nn.Module):
    def __init__(self, in_channels=99, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, batch_norm=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)

        self.resblock1 = ResidualBlock(512)
        self.resblock2 = ResidualBlock(512)
        self.resblock3 = ResidualBlock(512)
        self.resblock4 = ResidualBlock(512)
        self.resblock5 = ResidualBlock(512)
        self.resblock6 = ResidualBlock(512)
        self.resblock7 = ResidualBlock(512)
        self.resblock8 = ResidualBlock(512)
        self.resblock9 = ResidualBlock(512)  # (n, 512, 16, 16)

        self.up1 = UNetUp(512 + 512 + in_channels, 256, is_last=False)
        self.up2 = UNetUp(256 + 256 + in_channels, 128, is_last=False)
        self.up3 = UNetUp(128 + 128 + in_channels, 64, is_last=False)
        self.up4 = UNetUp(64 + 64 + in_channels, out_channels, is_last=True, batch_norm=False)

        self.final = nn.Sequential(nn.Tanh(),)

    def forward(self, x):
        """
        U-Net generator with skip connections from encoder to decoder.
        :param x: (n, 9Nw, h, w).
        :return: (n, 3, h, w).
        """

        d1 = self.down1(x)  # (n, 64, 128, 128)
        d2 = self.down2(d1)  # (n, 128, 64, 64)
        d3 = self.down3(d2)  # (n, 256, 32, 32)
        d4 = self.down4(d3)  # (n, 512, 16, 16)

        res1 = self.resblock1(d4)
        res2 = self.resblock2(res1)
        res3 = self.resblock3(res2)
        res4 = self.resblock4(res3)
        res5 = self.resblock5(res4)
        res6 = self.resblock6(res5)
        res7 = self.resblock7(res6)
        res8 = self.resblock8(res7)
        res9 = self.resblock9(res8)  # (n, 512, 16, 16)

        # Calculate the bilinear downsampling of input x.
        downsample_16 = bilinear_interpolation(x, 16, 16)  # (n, 9Nw, 16, 16). FloatTensor
        downsample_32 = bilinear_interpolation(x, 32, 32)  # (n, 9Nw, 32, 32). FloatTensor
        downsample_64 = bilinear_interpolation(x, 64, 64)  # (n, 9Nw, 64, 64). FloatTensor
        downsample_128 = bilinear_interpolation(x, 128, 128)  # (n, 9Nw, 128, 128). FloatTensor

        b = torch.cat((res9, d4, downsample_16), 1)  # (n, 512+512+9Nw, 16, 16)
        u1 = self.up1(b, skip_input=d3, downsample_input=downsample_32)  # (n, 256+256+9Nw, 32, 32)
        u2 = self.up2(u1, skip_input=d2, downsample_input=downsample_64)  # (n, 128+128+9Nw, 64, 64)
        u3 = self.up3(u2, skip_input=d1, downsample_input=downsample_128)  # (n, 64+64+9Nw, 128, 128)
        u4 = self.up4(u3)  # (n, 3, 256, 256)

        return self.final(u4)


class GeneratorFork(nn.Module):
    def __init__(self, in_channels=9, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, batch_norm=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512)

        self.resblock1 = ResidualBlock(128)
        self.resblock2 = ResidualBlock(128)
        self.resblock3 = ResidualBlock(128)
        self.resblock4 = ResidualBlock(128)
        self.resblock5 = ResidualBlock(128)
        self.resblock6 = ResidualBlock(128)
        self.resblock7 = ResidualBlock(128)
        self.resblock8 = ResidualBlock(128)
        self.resblock9 = ResidualBlock(128)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(512, 512, dropout=0.5)
        self.up3 = UNetUp(512, 512, dropout=0.5)
        self.up4 = UNetUp(512, 512)
        self.up5 = UNetUp(512, 256)
        self.up6 = UNetUp(256, 128)
        self.up7 = UNetUp(128 + 128 + in_channels, 64)
        self.up8 = UNetUp(64 + in_channels, out_channels, is_last=True, batch_norm=False)

        self.final = nn.Sequential(nn.Tanh(),)

    def forward(self, x):
        """
        Generator with fork structure.
        :param x: (n, 9, h, w).
        :return: (n, 3, h, w).
        """

        d1 = self.down1(x)  # (n, 64, 128, 128)
        d2 = self.down2(d1)  # (n, 128, 64, 64)
        # fork1
        d3 = self.down3(d2)  # (n, 256, 32, 32)
        d4 = self.down4(d3)  # (n, 512, 16, 16)
        d5 = self.down5(d4)  # (n, 512, 8, 8)
        d6 = self.down6(d5)  # (n, 512, 4, 4)
        d7 = self.down7(d6)  # (n, 512, 2, 2)
        d8 = self.down8(d7)  # (n, 512, 1, 1)
        u1 = self.up1(d8)  # (n, 512, 2, 2)
        u2 = self.up2(u1)  # (n, 512, 4, 4)
        u3 = self.up3(u2)  # (n, 512, 8, 8)
        u4 = self.up4(u3)  # (n, 512, 16, 16)
        u5 = self.up5(u4)  # (n, 256, 32, 32)
        u6 = self.up6(u5)  # (n, 128, 16, 16)

        # fork2
        res1 = self.resblock1(d2)
        res2 = self.resblock2(res1)
        res3 = self.resblock3(res2)
        res4 = self.resblock4(res3)
        res5 = self.resblock5(res4)
        res6 = self.resblock6(res5)
        res7 = self.resblock7(res6)
        res8 = self.resblock8(res7)
        res9 = self.resblock9(res8)  # (n, 128, 64, 64)

        # Calculate the bilinear downsampling of input x.
        downsample_64 = bilinear_interpolation(x, 64, 64)  # (n, 9Nw, 64, 64). FloatTensor
        downsample_128 = bilinear_interpolation(x, 128, 128)  # (n, 9Nw, 128, 128). FloatTensor

        b = torch.cat((u6, res9, downsample_64), 1)  # (n, 128+128+9Nw, 64, 64)
        u7 = self.up7(b, downsample_input=downsample_128)  # (n, 64+9Nw, 128, 128)
        u8 = self.up8(u7)  # (n, 3, 256, 256)

        return self.final(u8)


# class DiscriminatorTemp(nn.Module):
#     def __init__(self, in_channels=3):
#         super().__init__()
#
#         def discriminator_block3d(in_filters, out_filters, is_first=False, is_last=False):
#             """Returns down-sampling layers after each discriminator block."""
#             kernel_size = (3, 7, 7)
#             padding1 = (2, 3, 3)
#             padding2 = (1, 3, 3)
#             if is_first:
#                 layers = [
#                     nn.Conv3d(
#                         in_filters, out_filters, kernel_size=kernel_size, stride=2, padding=padding1
#                     )
#                 ]
#             else:
#                 layers = [
#                     nn.Conv3d(
#                         in_filters, out_filters, kernel_size=kernel_size, stride=1, padding=padding2
#                     )
#                 ]
#
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             layers.append(
#                 nn.Conv3d(
#                     out_filters, out_filters, kernel_size=kernel_size, stride=1, padding=padding2
#                 )
#             )
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             if not is_last:
#                 layers.append(torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
#             else:
#                 layers.append(nn.Conv3d(out_filters, 1, kernel_size=(3, 1, 1), stride=1, padding=0))
#             return layers
#
#         self.model = nn.Sequential(
#             *discriminator_block3d(in_channels, 32, is_first=True),
#             *discriminator_block3d(32, 64),
#             *discriminator_block3d(64, 128),
#             *discriminator_block3d(128, 256, is_last=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, *frames):
#         """Concatenate continuous frames by channels and feed to the D_tmp.
#
#         :param frames: a tuple of continuous frames, each frame has shape (n, 3, h, w) <Tensor + Variable>
#         :return: (n, 1, 1, h/16, w/16) <Tensor + Variable>
#         """
#         discriminator_input = frames[0].unsqueeze(2)  # (n, 3, 1, h, w)
#         for index, frame in enumerate(frames):
#             if index == 0:
#                 continue
#             discriminator_input = torch.cat(
#                 (discriminator_input, frame.unsqueeze(2)), 2
#             )  # (n, 3, n_frame, h, w)
#         return self.model(discriminator_input)
