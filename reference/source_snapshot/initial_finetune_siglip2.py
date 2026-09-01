"""Fine-tune SigLIP2-so400m on ALL 800 rail train images (same hyperparams as fold eval),
then extract features for 800 train + 300 test. Saves ft_siglip2_final.npz."""
import json, time, glob, os
import numpy as np
import pandas as pd
import torch, timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import warnings; warnings.filterwarnings("ignore")

DEV="mps"; IMG=378; EPOCHS=3
MODEL="vit_so400m_patch14_siglip_378.webli"
torch.manual_seed(0); np.random.seed(0)

lab = pd.read_csv("rail_labels.csv")
lab_map = dict(zip(lab["stem"], lab["defectType"]))
tr_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob("rail_img/rail_1024/train/*")}
te_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob("rail_img/rail_1024/test/*")}
tr_stems = sorted(lab_map.keys()); te_stems = sorted(te_paths.keys())
classes = sorted(set(lab_map.values())); cidx={c:i for i,c in enumerate(classes)}
y = np.array([cidx[lab_map[s]] for s in tr_stems])
print(f"train {len(tr_stems)} test {len(te_stems)}", flush=True)

model = timm.create_model(MODEL, pretrained=True, num_classes=len(classes), img_size=IMG).to(DEV)
cfg = timm.data.resolve_data_config({}, model=model)
mean,std = cfg["mean"], cfg["std"]
tf_train = T.Compose([T.Resize((IMG,IMG)), T.RandomHorizontalFlip(), T.ColorJitter(0.2,0.2,0.1), T.ToTensor(), T.Normalize(mean,std)])
tf_eval  = T.Compose([T.Resize((IMG,IMG)), T.ToTensor(), T.Normalize(mean,std)])

class DS(Dataset):
    def __init__(self, stems_, pathmap, tf, labels=None):
        self.stems, self.pm, self.tf, self.labels = stems_, pathmap, tf, labels
    def __len__(self): return len(self.stems)
    def __getitem__(self,k):
        s=self.stems[k]
        x=self.tf(Image.open(self.pm[s]).convert("RGB"))
        return (x, self.labels[k]) if self.labels is not None else (x, 0)

for p in model.parameters(): p.requires_grad=False
for blk in model.blocks[-4:]:
    for p in blk.parameters(): p.requires_grad=True
for p in model.norm.parameters(): p.requires_grad=True
for p in model.head.parameters(): p.requires_grad=True
opt = torch.optim.AdamW([
    {"params":[p for b in model.blocks[-4:] for p in b.parameters()],"lr":3e-5},
    {"params":list(model.norm.parameters())+list(model.head.parameters()),"lr":1e-3}],
    weight_decay=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
dl = DataLoader(DS(tr_stems, tr_paths, tf_train, y), batch_size=8, shuffle=True, num_workers=0)

for ep in range(EPOCHS):
    model.train(); t0=time.time(); tot=0; n=0
    for xb,yb in dl:
        opt.zero_grad()
        loss = crit(model(xb.to(DEV)), yb.to(DEV))
        loss.backward(); opt.step()
        tot+=loss.item()*len(yb); n+=len(yb)
    sched.step()
    print(f"ep{ep+1} loss {tot/n:.3f} ({time.time()-t0:.0f}s)", flush=True)

def extract(stems_, pm):
    model.eval(); fs=[]
    dl_ = DataLoader(DS(stems_, pm, tf_eval), batch_size=16, num_workers=0)
    with torch.no_grad():
        for xb,_ in dl_:
            f = model.forward_head(model.forward_features(xb.to(DEV)), pre_logits=True)
            fs.append(f.cpu().numpy())
    return np.concatenate(fs)

t0=time.time()
Ftr = extract(tr_stems, tr_paths); Fte = extract(te_stems, te_paths)
print(f"extracted {Ftr.shape} {Fte.shape} ({time.time()-t0:.0f}s)", flush=True)
np.savez("ft_siglip2_final.npz", tr_stems=tr_stems, te_stems=te_stems, Ftr=Ftr, Fte=Fte)
torch.save(model.state_dict(), "ft_siglip2_final.pt")
print("DONE", flush=True)
