import os
import torch
import urllib.request
from .restormer_arch import Restormer

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), '../../weights/restormer_real_denoising.pth')

def load_restormer_model(device="cuda", weights_path=DEFAULT_WEIGHTS):
    """Loads the Restormer model and downloads weights if missing."""
    model = Restormer(
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree'
    )
    
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    if not os.path.exists(weights_path):
        print(f"[Restormer] Downloading weights to {weights_path}...")
        url = "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(weights_path, 'wb') as out_file:
                out_file.write(response.read())
            print("[Restormer] Pretrained weights downloaded successfully!")
        except Exception as e:
            print(f"[Restormer] Failed to download weights: {e}")
            print(f"[Restormer] Please download 'real_denoising.pth' manually to {weights_path}")
            
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        if "params" in state_dict:
            state_dict = state_dict["params"]
        model.load_state_dict(state_dict, strict=True)
        
    model = model.to(device)
    model.eval()
    return model
