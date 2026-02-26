import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

HAM_CLASSES=['akiec','bcc','bkl','df','mel','nv','vasc']

def load_model(model_path='best_model.pth'):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try: model=mobilenet_v2(weights='DEFAULT')
    except: model=mobilenet_v2(pretrained=True)
    in_f=model.classifier[1].in_features
    model.classifier=nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, len(HAM_CLASSES)))
    ckpt=torch.load(model_path, map_location=device)
    if isinstance(ckpt,dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, device
