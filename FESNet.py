import torch.nn as nn
import torch
import torch.nn.functional as F
from ops import *


class Residual_block(nn.Module):
    def __init__(self,in_channels,out_channels,wn) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        if wn is not None:
            self.conv1 = wn(self.conv1)
            self.conv2 = wn(self.conv2)
        
    
    def forward(self,x):
        res = self.conv1(x)
        res = torch.nn.functional.relu(res)
        res = self.conv2(res)
        res = x+res
        res = torch.nn.functional.relu(res)
        return res

class ASPP_conv(nn.Sequential):
    def __init__(self,kernal,dilation,channal_out,channal_in,wn):
        K = kernal+(kernal-1)*(dilation-1)
        tmp = channal_out-1-channal_in+K
        assert tmp % 2 == 0,"The design of ASPP_conv have a problem,please redesign carefully" 
        padding = tmp//2
        
        modules = [
            wn(nn.Conv2d(in_channels=channal_in,out_channels=channal_out,kernel_size=kernal,stride=1,dilation=dilation,padding=padding)),
            nn.ReLU()
        ]    
        super(ASPP_conv, self).__init__(*modules)
    
class ASPP2_block(nn.Module):
    def __init__(self,wn) -> None:
        super().__init__()
        self.conv1 = ASPP_conv(channal_in=64,channal_out=64,kernal=3,dilation=1,wn=wn)#K = 3+2*0=3; out = 64-3+2+1=64
        self.conv2 = ASPP_conv(channal_in=64,channal_out=64,kernal=3,dilation=4,wn=wn)#K = 3+2*1=5; out = 64-5+2P+1=>P=2
        self.conv3 = ASPP_conv(channal_in=64,channal_out=64,kernal=3,dilation=8,wn=wn)#K = 3+2*3=9; out = 64-9+2P+1=>P=4
        self.conv4 = wn(nn.Conv2d(in_channels=64*3,out_channels=64,kernel_size=1,stride=1         ,padding=0))
    
    def forward(self,x):
        res = []
        res.append(self.conv1(x))
        res.append(self.conv2(x))
        res.append(self.conv3(x))
        res = torch.cat(res,dim=1)
        res = self.conv4(res)
        return res

class ASPP2(nn.Module):
    def __init__(self,wn) -> None:
        super().__init__()

        self.ASPP2_block1 = ASPP2_block(wn=wn)
        self.ASPP2_block2 = ASPP2_block(wn=wn)
        self.ASPP2_block3 = ASPP2_block(wn=wn)
        self.residual1 = Residual_block(in_channels=64,out_channels=64,wn=wn)


    def forward(self,x):
        res = self.ASPP2_block1(x)
        res1 = res+x
        
        res = self.ASPP2_block2(res)
        res2 = res+res1

        res = self.ASPP2_block3(res)
        res3 = res+res2

        res = self.residual1(res3)
        return res

class FENet(nn.Module):

    def __init__(self, **kwargs):
        super(FENet, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.n_blocks = 12
        scale = kwargs.get("scale")
        group = kwargs.get("group", 4)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)

        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))

        self.assp_residual1 = ASPP2(wn=wn)
        self.assp_residual2 = ASPP2(wn=wn)

        body = [FEB(wn, 64, 64) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)
        self.reduction = BasicConv2d(wn, 64*13, 64, 1, 1, 0)

        self.upscample = UpsampleBlock(64, scale=scale, multi_scale=False, wn=wn, group=group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

    def forward(self, x, scale):

        x = self.sub_mean(x)
        res = x
        x = self.entry_1(x)

        c0 = self.assp_residual1(x)
        c0 = x+c0
        c0 = self.assp_residual2(c0)
        
        out_blocks = []

        out_blocks.append(c0)

        for i in range(self.n_blocks):
            x = self.body[i](x)
            out_blocks.append(x)

        output = self.reduction(torch.cat(out_blocks, 1))

        output = output + x

        output = self.upscample(output, scale=scale)
        output = self.exit(output)

        skip  = F.interpolate(res, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)

        output = skip + output

        output = self.add_mean(output)

        return output


class Channal_Attention(nn.Module):
    def __init__(self,**kwargs) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ca_conv1 = nn.Conv2d(in_channels=64,out_channels=64//16,kernel_size=1,stride=1,padding=0)
        self.relu = nn.ReLU()
        self.ca_conv2 = nn.Conv2d(in_channels=64//16,out_channels=64,kernel_size=1,stride=1,padding=0)
        self.sigoid = nn.Sigmoid()
    
    def forward(self,x):
        res_ca = self.pool(x)
        res_ca = self.relu(self.ca_conv1(res_ca))
        a = self.sigoid(self.ca_conv2(res_ca))
        return torch.mul(a,x)


class Attention_Block(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.channal_attention = Channal_Attention()

        self.relu = nn.ReLU()
        self.sigoid = nn.Sigmoid()
        self.sa_conv1 = nn.Conv2d(in_channels=64,out_channels=64*2,kernel_size=1,stride=1,padding=0)
        self.sa_conv2 = nn.Conv2d(in_channels=64*2,out_channels=1,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        res = []
        
        res.append(self.channal_attention(x))

        res_sa = self.relu(self.sa_conv1(x))
        b = self.sigoid(self.sa_conv2(res_sa))
        res.append(torch.mul(b,x))

        return torch.cat(res, 1)



class LFM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.up_conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.act = nn.ReLU()
        self.up_conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)

        self.dn_conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.dn_conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)

        self.residual = Residual_block(in_channels=64,out_channels=64,wn = None)

        self.attention = Channal_Attention()


    
    def forward(self,x):
        res_up = self.up_conv1(x)
        res_up = self.act(res_up)
        res_up = self.up_conv2(res_up)

        res_dn = self.dn_conv1(x)
        res_dn = self.act(res_dn)
        res_dn = self.dn_conv2(res_dn)

        res = res_up+res_dn+x

        res = self.residual(res)

        res = self.attention(res)
        return res

class LFM_change_Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.up_conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.act = nn.ReLU()
        self.up_conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)

        self.dn_conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.dn_conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)

        self.residual = Residual_block(in_channels=64,out_channels=64,wn = None)
    
    def forward(self,x):
        res_up = self.up_conv1(x)
        res_up = self.act(res_up)
        res_up = self.up_conv2(res_up)

        res_dn = self.dn_conv1(x)
        res_dn = self.act(res_dn)
        res_dn = self.dn_conv2(res_dn)

        res = res_up+res_dn+x
        res = self.residual(res)
        return res

class LFM_Chain_change_Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lfm1 = LFM_change_Attention()
        self.conv1 = nn.Conv2d(in_channels=64*1,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.lfm2 = LFM_change_Attention()
        self.conv2 = nn.Conv2d(in_channels=64*2,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.lfm3 = LFM_change_Attention()
        self.conv3 = nn.Conv2d(in_channels=64*3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        res1 = self.lfm1(x)
        res = self.conv1(res1)

        res2 = self.lfm2(res)
        res_concat = torch.cat([res1,res2],1)
        res = self.conv2(res_concat)

        res3 = self.lfm3(res)
        res_concat = torch.cat([res_concat,res3],1)
        res = self.conv3(res_concat)
        return res

class LFM_Chain(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lfm1 = LFM()
        self.conv1 = nn.Conv2d(in_channels=64*1,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.lfm2 = LFM()
        self.conv2 = nn.Conv2d(in_channels=64*2,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.lfm3 = LFM()
        self.conv3 = nn.Conv2d(in_channels=64*3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        res1 = self.lfm1(x)
        res = self.conv1(res1)

        res2 = self.lfm2(res)
        res_concat = torch.cat([res1,res2],1)
        res = self.conv2(res_concat)

        res3 = self.lfm3(res)
        res_concat = torch.cat([res_concat,res3],1)
        res = self.conv3(res_concat)
        return res

class BP_Block(nn.Module):
    def __init__(self,**kwargs) -> None:
        super().__init__()
        scale = kwargs.get("scale")
        if scale == 2:
            self.kernel = 6 
            self.stride = 2
            self.padding = 2
        elif scale == 3:
            self.kernel = 9
            self.stride = 3
            self.padding = 3
        elif scale == 4:
            self.kernel = 8
            self.stride = 4
            self.padding = 2
        else:
            raise(NotImplementedError)
        self.dconv = nn.ConvTranspose2d(in_channels=64,out_channels=64, kernel_size=self.kernel,stride= self.stride,padding= self.padding)
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3,stride = 1,padding = 1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=self.kernel,stride= self.stride,padding= self.padding)
        self.relu = nn.ReLU()
        

    def forward(self,x):
        res = self.relu(self.dconv(x))
        res = self.relu(self.conv1(res))
        res = self.relu(self.conv2(res))
        return res

class HFM(nn.Module):
    def __init__(self,**kwargs) -> None:
        super().__init__()
        scale = kwargs.get("scale")
        self.bp1 = BP_Block(scale=scale)
        self.residual1 = Residual_block(in_channels=64,out_channels=64,wn=None)
        self.bp2 = BP_Block(scale=scale)
        self.residual2 = Residual_block(in_channels=64,out_channels=64,wn=None)
        self.bp3 = BP_Block(scale=scale)
        self.residual3 = Residual_block(in_channels=64,out_channels=64,wn=None)
        self.conv = nn.Conv2d(in_channels=64*4,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.attention = Attention_Block()
        self.relu = nn.ReLU()

    def forward(self,x):
        res1 = self.bp1(x)
        sub1 = res1-x
        res1 = self.residual1(sub1)

        res2 = self.bp2(res1)
        sub2 = res2-res1
        res2 = self.residual2(sub2)

        res3 = self.bp3(res2)
        sub3 = res3-res1
        res3 = self.residual3(sub3)

        res = self.relu(self.conv(torch.cat([x,res1,res2,res3],1)))#
        return self.attention(res)

class HFM_change_Attention(nn.Module):
    def __init__(self,**kwargs) -> None:
        super().__init__()
        scale = kwargs.get("scale")
        self.bp1 = BP_Block(scale=scale)
        self.residual1 = Residual_block(in_channels=64,out_channels=64,wn=None)
        self.bp2 = BP_Block(scale=scale)
        self.residual2 = Residual_block(in_channels=64,out_channels=64,wn=None)
        self.bp3 = BP_Block(scale=scale)
        self.residual3 = Residual_block(in_channels=64,out_channels=64,wn=None)
        self.conv = nn.Conv2d(in_channels=64*4,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        res1 = self.bp1(x)
        sub1 = res1-x
        res1 = self.residual1(sub1)

        res2 = self.bp2(res1)
        sub2 = res2-res1
        res2 = self.residual2(sub2)

        res3 = self.bp3(res2)
        sub3 = res3-res1
        res3 = self.residual3(sub3)

        res = self.relu(self.conv(torch.cat([x,res1,res2,res3],1)))
        return res

class HFM_Chain_change_Attention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        scale = kwargs.get("scale")
        self.hfm1 = HFM_change_Attention(scale=scale)
        self.conv1 = nn.Conv2d(in_channels=64*1,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.hfm2 = HFM_change_Attention(scale=scale)
        self.conv2 = nn.Conv2d(in_channels=64*2,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.hfm3 = HFM_change_Attention(scale=scale)
        self.conv3 = nn.Conv2d(in_channels=64*3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        res1 = self.hfm1(x)
        res = self.conv1(res1)

        res2 = self.hfm2(res)
        res_concat = torch.cat([res1,res2],1)
        res = self.conv2(res_concat)

        res3 = self.hfm3(res)
        res_concat = torch.cat([res_concat,res3],1)
        res = self.conv3(res_concat)
        return res

class HFM_Chain(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        scale = kwargs.get("scale")
        self.hfm1 = HFM(scale=scale)
        self.conv1 = nn.Conv2d(in_channels=64*2,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.hfm2 = HFM(scale=scale)
        self.conv2 = nn.Conv2d(in_channels=64*4,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.hfm3 = HFM(scale=scale)
        self.conv3 = nn.Conv2d(in_channels=64*6,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        res1 = self.hfm1(x)
        res = self.conv1(res1)

        res2 = self.hfm2(res)
        res_concat = torch.cat([res1,res2],1)
        res = self.conv2(res_concat)

        res3 = self.hfm3(res)
        res_concat = torch.cat([res_concat,res3],1)
        res = self.conv3(res_concat)
        return res

class DRFFM_not_l(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        scale = kwargs.get("scale")
        self.hfm_chain = HFM_Chain(scale=scale)

        self.conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        res = self.hfm_chain(x)
        return self.relu(self.conv(res))

class DRFFM_not_h(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lfm_chain = LFM_Chain()
        scale = kwargs.get("scale")
        self.conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        res = self.lfm_chain(x)
        return self.relu(self.conv(res))

class FSAB(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.lfm_chain = LFM_Chain()

        scale = kwargs.get("scale")
        self.hfm_chain = HFM_Chain(scale=scale)

        self.conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        res = self.lfm_chain(x)+self.hfm_chain(x)
        return self.relu(self.conv(res))

class DRFFM_change_Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.lfm_chain = LFM_Chain_change_Attention()

        scale = kwargs.get("scale")
        self.hfm_chain = HFM_Chain_change_Attention(scale=scale)

        self.conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        res = self.lfm_chain(x)+self.hfm_chain(x)
        return self.relu(self.conv(res))

class Multi_deep_branch_not_l(nn.Module):
    def __init__(self, **kwargs):
        super(Multi_deep_branch_not_l, self).__init__()
        self.scale = kwargs.get("scale")
        self.group = kwargs.get("group", 4)
        self.n_blocks = kwargs.get("n_blocks",1)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))
        self.assp_residual1 = ASPP2(wn=wn)

        self.assp_residual2 = ASPP2(wn=wn)

        body = [DRFFM_not_l(scale=self.scale) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)

        self.concat_conv = nn.Conv2d(in_channels=64*self.n_blocks,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upscample = UpsampleBlock(64, scale=self.scale, multi_scale=False, wn=wn, group=self.group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x):
        x = self.sub_mean(x)
        lr = x
        x = self.entry_1(x)

        c0 = self.assp_residual1(x)
        c0 = x+c0
        c0 = self.assp_residual2(c0)

        out_blocks = []
        
        for i in range(self.n_blocks):
            c0 = self.body[i](c0)
            out_blocks.append(c0)
        
        res = self.concat_conv(torch.cat(out_blocks,1))
        res = self.upscample(res, scale=self.scale)
        res = self.exit(res)

        skip  = F.interpolate(lr, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode='bicubic', align_corners=False)
        res = skip + res
        res = self.add_mean(res)
        return res

class Multi_deep_branch_not_h(nn.Module):
    def __init__(self, **kwargs):
        super(Multi_deep_branch_not_h, self).__init__()
        self.scale = kwargs.get("scale")
        self.group = kwargs.get("group", 4)
        self.n_blocks = kwargs.get("n_blocks",1)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))
        self.assp_residual1 = ASPP2(wn=wn)

        self.assp_residual2 = ASPP2(wn=wn)

        body = [DRFFM_not_h(scale=self.scale) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)

        self.concat_conv = nn.Conv2d(in_channels=64*self.n_blocks,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upscample = UpsampleBlock(64, scale=self.scale, multi_scale=False, wn=wn, group=self.group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x):
        x = self.sub_mean(x)
        lr = x
        x = self.entry_1(x)

        c0 = self.assp_residual1(x)
        c0 = x+c0
        c0 = self.assp_residual2(c0) 

        out_blocks = []
        
        for i in range(self.n_blocks):
            c0 = self.body[i](c0)
            out_blocks.append(c0)
        
        res = self.concat_conv(torch.cat(out_blocks,1))
        res = self.upscample(res, scale=self.scale)
        res = self.exit(res)
        skip  = F.interpolate(lr, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode='bicubic', align_corners=False)
        res = skip + res
        res = self.add_mean(res)
        return res


class Multi_deep_branch_change_ASPP(nn.Module):
    def __init__(self, **kwargs):
        super(Multi_deep_branch_change_ASPP, self).__init__()
        self.scale = kwargs.get("scale")
        self.group = kwargs.get("group", 4)
        self.n_blocks = kwargs.get("n_blocks",1)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))
        self.residual1 = Residual_block(in_channels=64,out_channels=64,wn=wn)
        self.residual2 = Residual_block(in_channels=64,out_channels=64,wn=wn)

        body = [FSAB(scale=self.scale) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)

        self.concat_conv = nn.Conv2d(in_channels=64*self.n_blocks,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upscample = UpsampleBlock(64, scale=self.scale, multi_scale=False, wn=wn, group=self.group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x):
        x = self.sub_mean(x)
        lr = x
        x = self.entry_1(x)

        c0 = self.residual1(x)
        c0 = self.residual2(c0)

        out_blocks = []
        
        for i in range(self.n_blocks):
            c0 = self.body[i](c0)
            out_blocks.append(c0)
        
        res = self.concat_conv(torch.cat(out_blocks,1))
        res = self.upscample(res, scale=self.scale)
        res = self.exit(res)

        skip  = F.interpolate(lr, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode='bicubic', align_corners=False)
        res = skip + res
        res = self.add_mean(res)
        return res

class Multi_deep_branch_change_Attention(nn.Module):
    def __init__(self, **kwargs):
        super(Multi_deep_branch_change_Attention, self).__init__()
        self.scale = kwargs.get("scale")
        self.group = kwargs.get("group", 4)
        self.n_blocks = kwargs.get("n_blocks",1)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))
        self.assp_residual1 = ASPP2(wn=wn)
        self.assp_residual2 = ASPP2(wn=wn)

        body = [DRFFM_change_Attention(scale=self.scale) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)

        self.concat_conv = nn.Conv2d(in_channels=64*self.n_blocks,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upscample = UpsampleBlock(64, scale=self.scale, multi_scale=False, wn=wn, group=self.group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x):
        x = self.sub_mean(x)
        lr = x
        x = self.entry_1(x)

        c0 = self.assp_residual1(x)
        c0 = x+c0
        c0 = self.assp_residual2(c0) 

        out_blocks = []
        
        for i in range(self.n_blocks):
            c0 = self.body[i](c0)
            out_blocks.append(c0)
        res = self.concat_conv(torch.cat(out_blocks,1))
        res = self.upscample(res, scale=self.scale)
        res = self.exit(res)

        skip  = F.interpolate(lr, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode='bicubic', align_corners=False)
        res = skip + res
        res = self.add_mean(res)
        return res

class FESNet(nn.Module):
    def __init__(self, **kwargs):
        super(FESNet, self).__init__()
        self.scale = kwargs.get("scale")
        self.group = kwargs.get("group", 4)
        self.n_blocks = kwargs.get("n_blocks",1)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))
        self.assp_residual1 = ASPP2(wn=wn)

        self.assp_residual2 = ASPP2(wn=wn)

        body = [FSAB(scale=self.scale) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)

        self.concat_conv = nn.Conv2d(in_channels=64*self.n_blocks,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upscample = UpsampleBlock(64, scale=self.scale, multi_scale=False, wn=wn, group=self.group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x):
        x = self.sub_mean(x)
        lr = x
        x = self.entry_1(x)

        c0 = self.assp_residual1(x)
        c0 = x+c0
        c0 = self.assp_residual2(c0)
        out_blocks = []
        
        for i in range(self.n_blocks):
            c0 = self.body[i](c0)
            out_blocks.append(c0)
        
        res = self.concat_conv(torch.cat(out_blocks,1))
        res = self.upscample(res, scale=self.scale)
        res = self.exit(res)

        skip  = F.interpolate(lr, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode='bicubic', align_corners=False)
        res = skip + res
        res = self.add_mean(res)
        return res


class DRFFM_change_LFAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.lfm_chain = LFM_Chain_change_Attention()

        scale = kwargs.get("scale")
        self.hfm_chain = HFM_Chain(scale=scale)

        self.conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        res = self.lfm_chain(x)+self.hfm_chain(x)
        return self.relu(self.conv(res))


class Multi_deep_branch_notLF_A(nn.Module):
    def __init__(self, **kwargs):
        super(Multi_deep_branch_notLF_A, self).__init__()
        self.scale = kwargs.get("scale")
        self.group = kwargs.get("group", 4)
        self.n_blocks = kwargs.get("n_blocks",1)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))
        self.assp_residual1 = ASPP2(wn=wn)

        self.assp_residual2 = ASPP2(wn=wn)

        body = [DRFFM_change_LFAttention(scale=self.scale) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)

        self.concat_conv = nn.Conv2d(in_channels=64*self.n_blocks,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upscample = UpsampleBlock(64, scale=self.scale, multi_scale=False, wn=wn, group=self.group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x):
        x = self.sub_mean(x)
        lr = x
        x = self.entry_1(x)

        c0 = self.assp_residual1(x)
        c0 = x+c0
        c0 = self.assp_residual2(c0)

        out_blocks = []
        
        for i in range(self.n_blocks):
            c0 = self.body[i](c0)
            out_blocks.append(c0)
        res = self.concat_conv(torch.cat(out_blocks,1))
        res = self.upscample(res, scale=self.scale)
        res = self.exit(res)
        skip  = F.interpolate(lr, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode='bicubic', align_corners=False)
        res = skip + res
        res = self.add_mean(res)
        return res


class DRFFM_change_HFAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.lfm_chain = LFM_Chain()
        scale = kwargs.get("scale")
        self.hfm_chain = HFM_Chain_change_Attention(scale=scale)
        self.conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        res = self.lfm_chain(x)+self.hfm_chain(x)
        return self.relu(self.conv(res))


class Multi_deep_branch_notHF_A(nn.Module):
    def __init__(self, **kwargs):
        super(Multi_deep_branch_notHF_A, self).__init__()
        self.scale = kwargs.get("scale")
        self.group = kwargs.get("group", 4)
        self.n_blocks = kwargs.get("n_blocks",1)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))
        self.assp_residual1 = ASPP2(wn=wn)

        self.assp_residual2 = ASPP2(wn=wn)

        body = [DRFFM_change_HFAttention(scale=self.scale) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)

        self.concat_conv = nn.Conv2d(in_channels=64*self.n_blocks,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.upscample = UpsampleBlock(64, scale=self.scale, multi_scale=False, wn=wn, group=self.group)
        self.exit = wn(nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x):
        x = self.sub_mean(x)
        lr = x
        x = self.entry_1(x)

        c0 = self.assp_residual1(x)
        c0 = x+c0
        c0 = self.assp_residual2(c0)

        out_blocks = []
        
        for i in range(self.n_blocks):
            c0 = self.body[i](c0)
            out_blocks.append(c0)
        res = self.concat_conv(torch.cat(out_blocks,1))
        res = self.upscample(res, scale=self.scale)
        res = self.exit(res)
        skip  = F.interpolate(lr, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode='bicubic', align_corners=False)
        res = skip + res
        res = self.add_mean(res)
        return res