import torch, numpy as np, cv2
from PIL import Image
import torchvision.transforms as T

transform=T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class GradCAM:
    def __init__(self, model, layer):
        self.model=model; self.layer=layer
        self.grad=None; self.act=None
        layer.register_forward_hook(self.fh)
        layer.register_backward_hook(self.bh)
    def fh(self, m,i,o): self.act=o.detach()
    def bh(self, m,gi,go): self.grad=go[0].detach()
    def __call__(self,x,cls):
        out=self.model(x); score=out[0,cls]
        self.model.zero_grad(); score.backward(retain_graph=True)
        g=self.grad[0].cpu().numpy(); a=self.act[0].cpu().numpy()
        w=g.mean(axis=(1,2)); cam=np.zeros(a.shape[1:],dtype=np.float32)
        for i,v in enumerate(w): cam+=v*a[i]
        cam=np.maximum(cam,0); cam=cv2.resize(cam,(224,224))
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cam

def predict_with_gradcam(pil, model, device, classes):
    x=transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits=model(x); probs=torch.softmax(logits,1)[0].cpu().numpy()
    idx=probs.argmax()
    layer=model.features[-1]
    cam=GradCAM(model,layer)(x,idx)
    orig=np.array(pil.resize((224,224)))
    heat=(cam*255).astype(np.uint8)
    heat=cv2.applyColorMap(heat,cv2.COLORMAP_JET)
    overlay=cv2.addWeighted(orig,0.6,heat,0.4,0)
    top=probs.argsort()[-3:][::-1]
    return [classes[i] for i in top],[float(probs[i]) for i in top],overlay,cam
