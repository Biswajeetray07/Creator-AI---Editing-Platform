import urllib.request
import os

weights_dir = r"c:\Users\raj08\OneDrive\Documents\Machine Models\Text Removal\weights"
os.makedirs(weights_dir, exist_ok=True)

urls = {
    'sam_vit_b_01ec64.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'modnet_photographic_portrait_matting.ckpt': 'https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.ckpt'
}

for name, url in urls.items():
    path = os.path.join(weights_dir, name)
    print(f"Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {name} successfully. Size: {os.path.getsize(path)} bytes")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
