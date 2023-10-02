from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F


@ARCH_REGISTRY.register()
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_channel=3, num_out_channel=3, num_feat=64, num_conv=16, upscale=2, act_type="prelu"):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_channel = num_in_channel
        self.num_out_channel = num_out_channel
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # The first conv
        self.body.append(nn.Conv2d(num_in_channel, num_feat, kernel_size=3, stride=1, padding=1))
        # The first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise NotImplementedError(f"This activation {act_type} does not implement. Please use relu, prelu or leakyrelu!")
        self.body.append(activation)
        # The body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1))
            # Activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            else:
                raise NotImplementedError(f"This activation {act_type} does not implement. Please use relu, prelu or leakyrelu!")
            self.body.append(activation)
        
        # The last conv
        self.body.append(nn.Conv2d(num_feat, num_in_channel * upscale * upscale, kernel_size=3, stride=1, padding=1))
        # Upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        # Add the nearest upsampled image 
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out += base
        return out



if __name__ == "__main__":
    import torch
    model = SRVGGNetCompact()
    x = torch.rand(1, 3, 224, 224)
    print(model(x).shape)