import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input_x):
        out = self.depth_conv(input_x)
        out = self.point_conv(out)
        return out

class ZeroDCEPlusPlus(nn.Module):
    def __init__(self):
        super(ZeroDCEPlusPlus, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32

        self.e_conv1 = DSConv(3, number_f)
        self.e_conv2 = DSConv(number_f, number_f)
        self.e_conv3 = DSConv(number_f, number_f)
        self.e_conv4 = DSConv(number_f, number_f)
        self.e_conv5 = DSConv(number_f * 2, number_f)
        self.e_conv6 = DSConv(number_f * 2, number_f)
        self.e_conv7 = DSConv(number_f * 2, 24)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        
        # Curve parameters (24 channels for 8 iterations * 3 RGB)
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        
        # Iterative curve application
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x)
        
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        return enhance_image

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), '../../weights/ZeroDCE++.pth')

def load_zero_dce_model(device="cuda", weights_path=DEFAULT_WEIGHTS):
    """Loads the ZeroDCE++ model and downloads weights if missing."""
    model = ZeroDCEPlusPlus()
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    # Check if weights exist, otherwise download from known mirror
    if not os.path.exists(weights_path):
        print(f"[Zero-DCE++] Downloading weights to {weights_path}...")
        url = "https://huggingface.co/IanNathaniel/Zero-DCE/resolve/main/Epoch99.pth"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(weights_path, 'wb') as out_file:
                out_file.write(response.read())
            print("[Zero-DCE++] Pretrained weights downloaded successfully!")
        except Exception as e:
            print(f"[Zero-DCE++] Failed to download weights: {e}")
            print("[Zero-DCE++] Please place standard ZeroDCE++ weights in weights directory.")
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Handle state dict wrapped in another dict (common in some repos)
        if hasattr(state_dict, "keys") and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Handle "module." prefix from DataParallel
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"[Zero-DCE++] Warning during weight load: {e}")
            
    model = model.to(device)
    model.eval()
    return model
