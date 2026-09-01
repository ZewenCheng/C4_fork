import torch, numpy as np, os, sys, time
from PIL import Image
dev = "mps"
spec = sys.argv[1]; tag = sys.argv[2]; SPLIT = sys.argv[3] if len(sys.argv)>3 else "train"
stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(f"rail_1024/{SPLIT}")) if f.endswith(".jpg")]

if spec.startswith("timm:"):
    import timm
    name = spec[5:]
    m = timm.create_model(name, pretrained=True, num_classes=0).to(dev).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = timm.data.create_transform(**cfg, is_training=False)
    def encode(ims):
        x = torch.stack([tf(i) for i in ims]).to(dev)
        with torch.no_grad(): return m(x).float().cpu().numpy()
else:
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(spec)
    mod = AutoModel.from_pretrained(spec)
    mod = getattr(mod, "vision_model", mod).to(dev).eval()
    def encode(ims):
        with torch.no_grad():
            x = proc(images=ims, return_tensors="pt").to(dev)
            o = mod(**x)
            f = getattr(o, "pooler_output", None)
            if f is None: f = o.last_hidden_state.mean(1)
            return f.float().cpu().numpy()

def views(im):
    w,h = im.size; s = min(w,h)
    return {"orig": im,
            "flip": im.transpose(Image.FLIP_LEFT_RIGHT),
            "crop": im.crop(((w-s)//2,(h-s)//2,(w-s)//2+s,(h-s)//2+s))}

out = {v: [] for v in ["orig","flip","crop"]}
t0 = time.time(); B = 8
for i in range(0, len(stems), B):
    batch = [Image.open(f"rail_1024/{SPLIT}/{s}.jpg").convert("RGB") for s in stems[i:i+B]]
    for v in out:
        out[v].append(encode([views(im)[v] for im in batch]))
    if i % 200 == 0: print(f"  {i}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
arr = {v: np.concatenate(out[v]) for v in out}
np.savez(f"feat_{tag}_{SPLIT}.npz", stems=np.array(stems), **arr)
print(f"{tag} done {time.time()-t0:.0f}s dim={arr['orig'].shape}")
