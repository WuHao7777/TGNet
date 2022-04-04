import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
affine_par = True
import functools

import sys, os

in_place = True

class FcAttentionGate(nn.Module):
    def __init__(self,Fg,Fl):
        super(FcAttentionGate,self).__init__()
        
        self.GAP=nn.Sequential(
            nn.GroupNorm(16,Fg),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool3d((1,1,1))
        )
        
        self.controller=nn.Conv3d(Fg+7,Fl,kernel_size=1,stride=1,padding=0)
    
    def forward(self,xl,g,task_encoding):
        g_feat=self.GAP(g)
        g_cond=torch.cat([g_feat,task_encoding],1)
        params=self.controller(g_cond)
        sigmoid=torch.sigmoid(params)
        attention=sigmoid.expand_as(xl)
        xl=xl*attention
        
        return xl

class AttentionGate(nn.Module):
    def __init__(self,Fg,Fl,Fint):
        super(AttentionGate,self).__init__()
        
        self.Wg=nn.Sequential(
            nn.GroupNorm(16,Fg),
            nn.ReLU(inplace=in_place),
            Conv3d(Fg,Fint,kernel_size=1,stride=1,padding=0,bias=False)
        )
        self.Wx=nn.Sequential(
            nn.GroupNorm(16,Fl),
            nn.ReLU(inplace=in_place),
            Conv3d(Fl,Fint,kernel_size=1,stride=2,padding=0,bias=False)
        )
        self.y=Conv3d(Fint,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.out=Conv3d(Fl,Fl,kernel_size=1,stride=1,padding=0)
        
    def forward(self,xl,g):
        xl_size_orig=xl.size()
        xl_=self.Wx(xl)
        g=self.Wg(g)
        relu=F.relu(xl_+g,inplace=in_place)
        y=self.y(relu)
        sigmoid=torch.sigmoid(y) 
        upsampled_sigmoid=F.interpolate(sigmoid,size=xl_size_orig[2:],mode='trilinear',align_corners=False)
        attention=upsampled_sigmoid.expand_as(xl)
        return self.out(attention*xl)
            
class DynAttentionGate(nn.Module):
    def __init__(self,Fg,Fl,Fint):
        super(DynAttentionGate,self).__init__()
        
        self.Wg=nn.Sequential(
            nn.GroupNorm(16,Fg),
            nn.ReLU(inplace=in_place),
            Conv3d(Fg,Fint,kernel_size=1,stride=1,padding=0,bias=False)
        )
        self.Wx=nn.Sequential(
            nn.GroupNorm(16,Fl),
            nn.ReLU(inplace=in_place),
            Conv3d(Fl,Fint,kernel_size=1,stride=2,padding=0,bias=False)
        )
        self.out=Conv3d(Fl,Fl,kernel_size=1,stride=1,padding=0)

    def DynAtt_forward(self,features,weights,biases,num_insts):
        assert features.dim()==5
        x=features
        x=F.conv3d(x,weights,bias=biases,stride=1,padding=0,
                   groups=num_insts)
        return x 
    
    def forward(self,xl,g,weight,bias):
        xl_size_orig=xl.size()
        N,_,D,H,W=xl.size()
        xl_=self.Wx(xl)
        g=self.Wg(g)
        relu=F.relu(xl_+g,inplace=in_place)
        features=relu
        N, _, D, H, W=features.size()
        features=features.reshape(1,-1,D,H,W)
        logits=self.DynAtt_forward(features,weight,bias,N)
        logits=logits.reshape(-1,1,D,H,W)
        sigmoid=torch.sigmoid(logits)
        upsampled_sigmoid=F.interpolate(sigmoid,size=xl_size_orig[2:],mode='trilinear',align_corners=False)
        attention=upsampled_sigmoid.expand_as(xl)
        return self.out(attention*xl)

    
    
class Conv3d(nn.Conv3d): # weight_std may decrease the range of output

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        '''
        For weight.mean(dim=1,keepdim=True), before:
        tensor([[[0, 1],
                 [2, 3]],
                [[4, 5],
                 [6, 7]]])
torch.Size([2, 2, 2])
          
          After:
          tensor([[[1., 2.]],
                  [[5., 6.]]])
torch.Size([2, 1, 2])
       weights.shape=(out_channels,in_channels/groups,kernel_size[0],kernel_size[1],kernel_size[2])
       and After the whole change ,it becomes (out_channel,1,1,1,1)
        '''
        weight = weight - weight_mean #weight still has the same shape, element-wise subtact the mean value of each out_channel
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1) #weight.view(weight.size(0),-1).shape=(out_channels, a*b*c*d).
        #after torch.var, it becomes shape=(out_channels) , and the value is sample Var  (/(N-1))
        # + is element-wise addition, torch.sqrt(a,dim=1) get the sqrt of each element in a, value=(sqrt(value1),sqrt(value2),....) shape=(out_channels)
        #after view(-1,1,1,1,1) now shape is (out_channels,1,1,1,1)
        weight = weight / std.expand_as(weight) #all elements of weight divided by its corresponding std_  var
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



def conv3x3x3(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)


class DynNoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, Fg,Fl, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(DynNoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, inplanes) # torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
        
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.relu = nn.ReLU(inplace=in_place) # An in-place operation is an operation that changes directly the content of a given Tensor without making a copy

        self.gn2 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.downsample = downsample # seems for processing residual
        self.dilation = dilation
        self.stride = stride
        
        self.GAP=nn.Sequential(
            nn.GroupNorm(16,Fg),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool3d((1,1,1))
        )
        
        self.controller=nn.Conv3d(Fg+7,Fl,kernel_size=1,stride=1,padding=0)
    def forward(self, x,task_encoding):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)
    
        out_feat=self.GAP(out)
        out_cond=torch.cat([out_feat,task_encoding],1)
        params=self.controller(out_cond)
        sigmoid=torch.sigmoid(params)
        attention=sigmoid.expand_as(residual)    

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        residual=residual*attention

        out = out + residual

        return out

    
    


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, inplanes) # torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
        '''
        The input channels are separated into num_groups groups, each containing num_channels / num_groups channels. The mean and standard-deviation are calculated separately over the each group. γ and β are learnable per-channel affine transform parameter vectors of size num_channels if affine is True. The standard-deviation is calculated via the biased estimator, equivalent to torch.var(input, unbiased=False).
'''
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.relu = nn.ReLU(inplace=in_place) # An in-place operation is an operation that changes directly the content of a given Tensor without making a copy

        self.gn2 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.downsample = downsample # seems for processing residual
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)


        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class unet3D(nn.Module):
    def __init__(self, layers, num_classes=3, weight_std = False):
        self.inplanes = 128
        self.weight_std = weight_std
        super(unet3D, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))
       
        self.layer0b = DynNoBottleneck(32, 32, downsample=None, fist_dilation=1, multi_grid=1, weight_std=self.weight_std,Fg=32,Fl=32)
        
        self.layer1a = self._make_layer(NoBottleneck, 32, 64, 1, stride=(2, 2, 2))
        self.layer1b = DynNoBottleneck(64, 64, downsample=None, fist_dilation=1, multi_grid=1, weight_std=self.weight_std,Fg=64,Fl=64)
        
        self.layer2a = self._make_layer(NoBottleneck, 64, 128, 1, stride=(2, 2, 2))
        self.layer2b = DynNoBottleneck(128, 128, downsample=None, fist_dilation=1, multi_grid=1, weight_std=self.weight_std,Fg=128,Fl=128)
        
        self.layer3a = self._make_layer(NoBottleneck, 128, 256, 1, stride=(2, 2, 2))
        self.layer3b = DynNoBottleneck(256, 256, downsample=None, fist_dilation=1, multi_grid=1, weight_std=self.weight_std,Fg=256,Fl=256)
        
        self.layer4a = self._make_layer(NoBottleneck, 256, 256, 1, stride=(2, 2, 2))
        self.layer4b = DynNoBottleneck(256, 256, downsample=None, fist_dilation=1, multi_grid=1, weight_std=self.weight_std,Fg=256,Fl=256)

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, 8, kernel_size=1)
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.controller = nn.Conv3d(256+7, 162, kernel_size=1, stride=1, padding=0)
        self.DynAttController=nn.Conv3d(7,484,kernel_size=1,stride=1,padding=0)
            
        self.DynAtt3=DynAttentionGate(256,256,256)
        self.DynAtt2=DynAttentionGate(128,128,128)
        self.DynAtt1=DynAttentionGate(64,64,64)
        self.DynAtt0=DynAttentionGate(32,32,32)
        
        self.FcAtt3=FcAttentionGate(256,256)
        self.FcAtt2=FcAttentionGate(128,128)
        self.FcAtt1=FcAttentionGate(64,64)
        self.FcAtt0=FcAttentionGate(32,32)

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1  #generate_multi_grid is a function, index and grids are 2 inputs, a%b= remainder
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1 #task_encoding.shape=(N,7)
        return task_encoding.cuda()
    
    def parse_att_params(self,params,channels,weight_nums,bias_nums):
        assert params.dim()==2
        assert len(weight_nums)==len(bias_nums)
        assert params.size(1)==sum(weight_nums)+sum(bias_nums)
        
        num_insts=params.size(0)
        params_splits=list(torch.split_with_sizes(params,weight_nums+bias_nums,dim=1))
        weight_splits=params_splits[0]
        bias_splits=params_splits[1]
        weight_splits=weight_splits.reshape(num_insts,-1,1,1,1)
        bias_splits=bias_splits.reshape(num_insts)
        return weight_splits,bias_splits
        
    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2 # params.shape=(N,162) eg a.shape=(1,2,3,4,5) , a.dim()=5
        #weights_nums=[64,64,16],bias_nums=[8,8,2]
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        )) # weight_nums+bias_nums=[64,64,16,8,8,2]
          # len(params_splits=6, params_splits[0].shape=(2,64)
            # [1]=(2,64), [2]=(2,16), [3]=(2,8),[4]=(2,8),[5]=(2,2)
        weight_splits = params_splits[:num_layers] # first 3 parts
        bias_splits = params_splits[num_layers:] # last 3 parts

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)
        #len(weight_splits)=3, weight_splits[0].shape=(16,8,1,1,1),weight_split[2].shape=[4,8,1,1,1]. bias_splits[0].shape=(16),
            # bias_splits[2].shape=(4)
        return weight_splits, bias_splits
    
    def DynAtt_forward(self,features,weights,biases,num_insts):
        assert features.dim()==5
        x=features
        x=F.conv3d(x,weights,bias=biases,stride=1,padding=0,
                   groups=num_insts)
        return x
        
    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, input, task_id):
        
        task_encoding = self.encoding_task(task_id)
        task_encoding.unsqueeze_(2).unsqueeze_(2).unsqueeze_(2) #.unsqueeze_(2).shape=(N,7,1) so 3times it is (2,7,1,1,1)        

        x = self.conv1(input)
        x = self.layer0b(x,task_encoding)
        skip0 = x

        x = self.layer1a(x)
        x = self.layer1b(x,task_encoding)
        skip1 = x

        x = self.layer2a(x)
        x = self.layer2b(x,task_encoding)
        skip2 = x

        x = self.layer3a(x)
        x = self.layer3b(x,task_encoding)
        skip3 = x

        x = self.layer4a(x)
        x = self.layer4b(x,task_encoding)
    
        x = self.fusionConv(x)

        xl3=x
        
        # generate conv filters for classification layer
        x_feat = self.GAP(x)
        x_cond = torch.cat([x_feat, task_encoding], 1)
        params = self.controller(x_cond)
        params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
            
        
        #get skip3
        skip3=self.FcAtt3(skip3,xl3,task_encoding)
        
        
        # x8
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)
        xl2=x
        skip2=self.FcAtt2(skip2,xl2,task_encoding)
        
        # x4
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)
        xl1=x
        skip1=self.FcAtt1(skip1,xl1,task_encoding)
        
        # x2
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)
        xl0=x
        skip0=self.FcAtt0(skip0,xl0,task_encoding)
        
        # x1
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)

        head_inputs = self.precls_conv(x)

        N, _, D, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, D, H, W) #now shape=(1,N*_,D,H,W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)
        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)

        logits = logits.reshape(-1, 2, D, H, W)

        return logits

def UNet3D(num_classes=1, weight_std=False):
    print("Using DynConv 8,8,2")
    model = unet3D([1, 2, 2, 2, 2], num_classes, weight_std)
    return model