import torch
import torch.nn as nn
from thop import profile

# 导入您的CSP-MGDK模块（复制您之前提供的代码）
class FGM(nn.Module):
    """恢复小目标优化的FGM - 包含高频增强"""
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))
        self.high_freq_boost = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)

    def forward(self, x):
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm='backward')
        
        H, W = x2_fft.shape[-2:]
        freq_h = torch.fft.fftfreq(H, device=x.device).view(-1, 1)
        freq_w = torch.fft.fftfreq(W, device=x.device).view(1, -1)
        freq_magnitude = torch.sqrt(freq_h**2 + freq_w**2)
        
        high_freq_mask = (freq_magnitude > 0.3).float()
        high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)
        
        enhanced_fft = x2_fft * (1.0 + self.high_freq_boost * high_freq_mask)
        out = x1 * enhanced_fft
        
        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)
        return out * self.alpha + x * self.beta

class OmniKernel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        
        self.dilated_horizontal = nn.Conv2d(
            dim, dim, kernel_size=(1, 7), 
            padding=(0, 33), dilation=(1, 11), 
            groups=dim, bias=False
        )
        
        self.dilated_vertical = nn.Conv2d(
            dim, dim, kernel_size=(7, 1),
            padding=(33, 0), dilation=(11, 1),
            groups=dim, bias=False
        )
        
        self.dilated_2d_pyramid = nn.ModuleList([
            nn.Conv2d(dim, dim//4, 3, padding=1, dilation=1, groups=dim//4),
            nn.Conv2d(dim, dim//4, 3, padding=6, dilation=6, groups=dim//4),
            nn.Conv2d(dim, dim//4, 3, padding=12, dilation=12, groups=dim//4),
            nn.Conv2d(dim, dim//4, 3, padding=20, dilation=20, groups=dim//4),
        ])
        
        self.pyramid_fusion = nn.Conv2d(dim, dim, 1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        
        self.continuity_repair = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
        ])
        
        self.branch_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, 5, 1),
            nn.Sigmoid()
        )
        
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)
        
        self.visdrone_detector = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.register_buffer('base_weights', torch.tensor([3.5, 3.5, 0.8, 0.1, 1.0]))
        
    def forward(self, x):
        out = self.in_conv(x)
        
        # FCA
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)
        
        # SCA
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        x_sca = self.fgm(x_sca)
        
        # 分支处理
        horizontal_out = self.dilated_horizontal(out)
        horizontal_out = self.continuity_repair[0](horizontal_out)
        
        vertical_out = self.dilated_vertical(out)
        vertical_out = self.continuity_repair[1](vertical_out)
        
        pyramid_features = []
        for pyramid_conv in self.dilated_2d_pyramid:
            feat = pyramid_conv(out)
            pyramid_features.append(feat)
        
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        pyramid_out = self.pyramid_fusion(pyramid_concat)
        pyramid_out = self.continuity_repair[2](pyramid_out)
        
        point_out = self.dw_11(out)
        
        branch_weights = self.branch_attention(out)
        edge_strength = self.visdrone_detector(out)
        base_w = torch.softmax(self.base_weights, dim=0)
        
        adaptive_w = branch_weights.squeeze(-1).squeeze(-1)
        combined_w = base_w.unsqueeze(0) * adaptive_w
        combined_w = torch.softmax(combined_w, dim=1)
        
        out = (x + 
               combined_w[:, 0:1, None, None] * (1 + edge_strength) * horizontal_out +
               combined_w[:, 1:2, None, None] * (1 + edge_strength) * vertical_out +
               combined_w[:, 2:3, None, None] * (1 - 0.3 * edge_strength) * pyramid_out +
               combined_w[:, 3:4, None, None] * point_out +
               combined_w[:, 4:5, None, None] * x_sca)
        
        out = self.act(out)
        return self.out_conv(out)

class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = nn.Conv2d(dim, dim, 1)
        self.cv2 = nn.Conv2d(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), 
                                        [int(x.size(1) * self.e), 
                                         int(x.size(1) * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

# 创建简单的测试模型来模拟您的实验
class SimpleBackbone(nn.Module):
    """简单的backbone模拟"""
    def __init__(self, dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim//4, 3, 2, 1)
        self.conv2 = nn.Conv2d(dim//4, dim//2, 3, 2, 1)
        self.conv3 = nn.Conv2d(dim//2, dim, 3, 2, 1)
        
    def forward(self, x):
        x1 = self.conv1(x)  # /2
        x2 = self.conv2(x1) # /4
        x3 = self.conv3(x2) # /8
        return [x1, x2, x3]

class ModelExp3(nn.Module):
    """实验3: Baseline + CSP-MGDK"""
    def __init__(self):
        super().__init__()
        self.backbone = SimpleBackbone(256)
        
        # 在多个位置使用CSP-MGDK (模拟您可能的配置)
        self.csp1 = CSPOmniKernel(64)   # 在较大特征图上使用
        self.csp2 = CSPOmniKernel(128)  # 在中等特征图上使用 
        self.csp3 = CSPOmniKernel(256)  # 在小特征图上使用
        
    def forward(self, x):
        features = self.backbone(x)
        
        # 在多个层级应用CSP-MGDK
        f1 = self.csp1(features[0])  # 80x80
        f2 = self.csp2(features[1])  # 40x40
        f3 = self.csp3(features[2])  # 20x20
        
        return [f1, f2, f3]

class ModelExp4(nn.Module):
    """实验4: RXFA + CSP-MGDK"""
    def __init__(self):
        super().__init__()
        # 更复杂的backbone (模拟RXFA)
        self.backbone = SimpleBackbone(256)
        
        # 额外的重参数化层
        self.reparam_conv1 = nn.Conv2d(64, 64, 7, 1, 3, groups=64)
        self.reparam_conv2 = nn.Conv2d(128, 128, 7, 1, 3, groups=128)
        
        # 只在关键位置使用CSP-MGDK
        self.csp = CSPOmniKernel(256)  # 只在最小的特征图上使用
        
    def forward(self, x):
        features = self.backbone(x)
        
        # 重参数化处理
        f1 = self.reparam_conv1(features[0]) + features[0]  # 80x80
        f2 = self.reparam_conv2(features[1]) + features[1]  # 40x40
        
        # 只在最后使用CSP-MGDK
        f3 = self.csp(features[2])  # 20x20
        
        return [f1, f2, f3]

def detailed_analysis():
    """详细分析两个模型"""
    print("创建模型...")
    model_exp3 = ModelExp3().cuda()
    model_exp4 = ModelExp4().cuda()
    
    input_tensor = torch.randn(1, 3, 640, 640).cuda()
    
    print("\n" + "="*60)
    print("模型对比分析")
    print("="*60)
    
    # 分析模型3
    print("\n实验3分析: Baseline + CSP-MGDK")
    print("-" * 40)
    try:
        flops_3, params_3 = profile(model_exp3, inputs=(input_tensor,))
        print(f"FLOPs: {flops_3/1e9:.2f}G")
        print(f"参数: {params_3/1e6:.2f}M")
    except Exception as e:
        print(f"FLOPs计算出错: {e}")
        params_3 = sum(p.numel() for p in model_exp3.parameters())
        print(f"参数: {params_3/1e6:.2f}M")
        flops_3 = 0
    
    # 统计CSP模块
    csp_count_3 = sum(1 for n, m in model_exp3.named_modules() if 'CSP' in type(m).__name__)
    omni_count_3 = sum(1 for n, m in model_exp3.named_modules() if 'OmniKernel' in type(m).__name__)
    fgm_count_3 = sum(1 for n, m in model_exp3.named_modules() if 'FGM' in type(m).__name__)
    
    print(f"CSP模块数量: {csp_count_3}")
    print(f"OmniKernel数量: {omni_count_3}")
    print(f"FGM模块数量: {fgm_count_3}")
    
    # 分析模型4
    print("\n实验4分析: RXFA + CSP-MGDK")
    print("-" * 40)
    try:
        flops_4, params_4 = profile(model_exp4, inputs=(input_tensor,))
        print(f"FLOPs: {flops_4/1e9:.2f}G")
        print(f"参数: {params_4/1e6:.2f}M")
    except Exception as e:
        print(f"FLOPs计算出错: {e}")
        params_4 = sum(p.numel() for p in model_exp4.parameters())
        print(f"参数: {params_4/1e6:.2f}M")
        flops_4 = 0
    
    csp_count_4 = sum(1 for n, m in model_exp4.named_modules() if 'CSP' in type(m).__name__)
    omni_count_4 = sum(1 for n, m in model_exp4.named_modules() if 'OmniKernel' in type(m).__name__)
    fgm_count_4 = sum(1 for n, m in model_exp4.named_modules() if 'FGM' in type(m).__name__)
    
    print(f"CSP模块数量: {csp_count_4}")
    print(f"OmniKernel数量: {omni_count_4}")
    print(f"FGM模块数量: {fgm_count_4}")
    
    # 对比结果
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    if flops_3 > 0 and flops_4 > 0:
        print(f"FLOPs差异: {(flops_3 - flops_4)/1e9:.2f}G")
        print(f"实验3比实验4多: {((flops_3 - flops_4)/flops_4)*100:.1f}%")
    
    print(f"参数差异: {(params_3 - params_4)/1e6:.2f}M")
    print(f"CSP模块差异: 实验3有{csp_count_3}个, 实验4有{csp_count_4}个")
    
    # 特征图尺寸分析
    print(f"\n特征图尺寸分析:")
    print(f"640x640输入 -> 80x80特征图: FFT复杂度约 {80*80*torch.log2(torch.tensor(80*80, dtype=torch.float)):.0f}")
    print(f"640x640输入 -> 40x40特征图: FFT复杂度约 {40*40*torch.log2(torch.tensor(40*40, dtype=torch.float)):.0f}")
    print(f"640x640输入 -> 20x20特征图: FFT复杂度约 {20*20*torch.log2(torch.tensor(20*20, dtype=torch.float)):.0f}")
    
    print(f"\n结论:")
    print(f"- 实验3在多个层级(大特征图)使用CSP-MGDK，FFT计算量大")
    print(f"- 实验4只在小特征图使用CSP-MGDK，FFT计算量小")
    print(f"- 这解释了为什么实验3的GFLOPs可能比实验4高")

if __name__ == "__main__":
    detailed_analysis()