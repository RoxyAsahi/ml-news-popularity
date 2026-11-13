"""Stream-convert JEV-9B: per shard download, merge the System-1 LoRA (alpha/r = 2), int8 per-row quantize linear weights,
drop lm_head, delete the raw shard. Output /home/user/jev/q/<shard>."""
import os, json, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
R = "autotrust/AutoTrust-Decider-Mini-9B"; RAW = "/home/user/jev/raw"
idx = json.load(open(f"{RAW}/model.safetensors.index.json"))["weight_map"]
ad = {}
with safe_open(f"{RAW}/adapter/adapter_model.safetensors", "pt") as f:
    for k in f.keys():
        ad[k.replace("base_model.model.", "")] = f.get_tensor(k)
cfg = json.load(open(f"{RAW}/adapter/adapter_config.json")); scale = cfg["lora_alpha"] / cfg["r"]
merged = 0
for shard in sorted(set(idx.values())):
    out = f"/home/user/jev/q/{shard}"
    if os.path.exists(out):
        continue
    p = hf_hub_download(R, shard, local_dir=RAW)
    T = {}
    with safe_open(p, "pt") as f:
        for k in f.keys():
            if k == "lm_head.weight":
                continue
            w = f.get_tensor(k)
            base = k[:-len(".weight")]
            if f"{base}.lora_A.weight" in ad:
                w = (w.float() + scale * ad[f"{base}.lora_B.weight"].float() @ ad[f"{base}.lora_A.weight"].float())
                merged += 1
            if k.startswith("model.layers.") and w.dim() == 2 and k.endswith("proj.weight"):
                w = w.float(); s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
                T[k + ".q"] = torch.round(w / s).to(torch.int8).contiguous(); T[k + ".s"] = s.to(torch.bfloat16).contiguous()
            else:
                T[k] = w.to(torch.bfloat16).contiguous()
    save_file(T, out); os.remove(p)
    print(shard, "done; merged so far", merged, flush=True)
print("all done; LoRA modules merged:", merged, "of", len(ad) // 2)
