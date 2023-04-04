import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .genotypes import *

# all node operations has two input and one output, 2C -> C
STEP_STEP_OPS = {
    'Sum': lambda C: Sum(),
    'ECAAttn': lambda C: ECAAttn(C),
    'ShuffleAttn': lambda C: ShuffleAttn(C),
    'ConcatConv': lambda C: ConcatConv(C)
}

class Sum(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return x + y

###################################################################################################
################################### ECAAttn Block #################################################

################################### Channel Attention Block #######################################

class channel_attention_block(nn.Module):

    """ Implements a Channel Attention Block """

    def __init__(self,in_channels):

        super(channel_attention_block, self).__init__()
        
        adaptive_k = self.channel_att_kernel_calc(in_channels)
        

        self.pool_types = ["max","avg"]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Conv1d(1,1,kernel_size=adaptive_k,padding=(adaptive_k-1)//2,bias=False)
        
        self.combine = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        channel_att_sum = None

        for pool_type in self.pool_types:

            if pool_type == "avg":

                avg_pool = self.avg_pool(x)
                channel_att_raw = self.conv(avg_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            elif pool_type == "max":

                max_pool = self.max_pool(x)
                channel_att_raw = self.conv(max_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            if channel_att_sum is None:

                channel_att_sum = channel_att_raw

            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        gate = self.sigmoid(channel_att_sum).expand_as(x)

        return self.combine(x*gate)
    
    
    def channel_att_kernel_calc(self,num_channels,gamma=2,b=1):
        b=1
        gamma = 2
        t = int(abs((math.log(num_channels,2)+b)/gamma))
        k = t if t%2 else t+1
        
        return k


################################### Spatial Attention Block #####################################


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )



class spatial_attention_block(nn.Module):

    """ Implements a Spatial Attention Block """

    def __init__(self):

        super(spatial_attention_block,self).__init__()

        kernel_size = 7

        self.compress = ChannelPool()

        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x_compress = self.compress(x)

        x_out = self.spatial(x_compress)

        gate = self.sigmoid(x_out)

        return x*gate

################################### Complete Attention Block#####################################

class ECAAttn(nn.Module):

    def __init__(self,C):

        super(ECAAttn,self).__init__()

        self.channel_attention_block = channel_attention_block(in_channels=2*C)

        self.spatial_attention_block = spatial_attention_block()

        self.relu = nn.ReLU()

    def forward(self,x,y):

        comb_in = torch.cat((x, y), dim=1)

        x_out = self.channel_attention_block(comb_in)
        x_out_1 = self.spatial_attention_block(x_out)
        x_out_2 = self.relu(x_out_1)

        return x_out_2

#######################################################################################################
################################### ShuffleAttn Block #################################################

class ShuffleAttn(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, C, groups=32):
        super(ShuffleAttn, self).__init__()
        self.C = 2*C
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.parameter.Parameter(torch.zeros(1, self.C // (2 * groups), 1, 1))
        self.cbias =  nn.parameter.Parameter(torch.ones(1, self.C // (2 * groups), 1, 1))
        self.sweight =  nn.parameter.Parameter(torch.zeros(1, self.C // (2 * groups), 1, 1))
        self.sbias =  nn.parameter.Parameter(torch.ones(1, self.C // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(self.C // (2 * groups), self.C // (2 * groups))
        self.combine = nn.Conv2d(self.C, int(self.C/2), kernel_size=1)
        self.relu = nn.ReLU()
        

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x, y):
        
        comb_in = torch.cat((x, y), dim=1)
        
        b, c, h, w = comb_in.shape

        comb_in = comb_in.reshape(b * self.groups, -1, h, w)

        x_0, x_1 = comb_in.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)

        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        # Reduce the Channels
        out = self.combine(out)

        # Activation

        out2 = self.relu(out)
        
        return out2
    

class ConcatConv(nn.Module):
    def __init__(self, C):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Conv2d(2*C, C, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)

        # Activation

        out2 = self.relu(out)
        
        return out2


class NodeMixedOp(nn.Module):
    def __init__(self, C, args):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in STEP_STEP_PRIMITIVES:
            op = STEP_STEP_OPS[primitive](C)
            self._ops.append(op)

    def forward(self, x, y, weights):
        out = sum(w * op(x, y) for w, op in zip(weights, self._ops))
        return out