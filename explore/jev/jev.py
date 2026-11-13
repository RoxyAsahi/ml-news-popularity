"""JEV-9B System 1 on CPU from the int8 shards made by convert.py (LoRA merged). decide(kind, items) -> probabilities."""
import json, glob, time, torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
RAW = "/home/user/jev/raw"
torch.set_num_threads(4)


class QLinear(nn.Module):
    def __init__(self, q, s):
        super().__init__(); self.q = q; self.s = s

    def forward(self, x):
        return F.linear(x, self.q.to(x.dtype) * self.s.to(x.dtype))


def load():
    cfg = AutoConfig.from_pretrained(RAW)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    with torch.device("meta"):
        model = M.Qwen3_5TextModel(tcfg)
    sd = {}
    for f in sorted(glob.glob("/home/user/jev/q/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                sd[k] = h.get_tensor(k)
    nq = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"model.{name}.{cname}" if name else f"model.{cname}"
            if isinstance(child, nn.Linear) and f"{full}.weight.q" in sd:
                setattr(mod, cname, QLinear(sd.pop(f"{full}.weight.q"), sd.pop(f"{full}.weight.s"))); nq += 1
    rest = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(rest, strict=False, assign=True)
    missing = [m for m in missing if not m.endswith((".q", ".s"))]
    model = model.to(torch.bfloat16).eval()
    # rotary buffers created on meta: re-init
    for m in model.modules():
        if hasattr(m, "inv_freq") and m.inv_freq.is_meta:
            inv, sc = m.compute_default_rope_parameters(m.config)
            m.inv_freq = inv; m.original_inv_freq = inv.clone(); m.attention_scaling = sc
    print("quantized linears", nq, "missing", missing[:5], len(missing), "unexpected", unexpected[:5], flush=True)
    return model


tok = AutoTokenizer.from_pretrained(RAW)
head = {}
with safe_open(f"{RAW}/head.safetensors", "pt") as h:
    W, b = h.get_tensor("proj.weight").float(), h.get_tensor("proj.bias").float()
CFG = json.load(open(f"{RAW}/judge_config.json")); TEMP = json.load(open(f"{RAW}/calibration.json"))["per_kind"]


def text(kind, state, question, options):
    lines = options if kind != "choice" else [f"{'ABCDEFGHIJKLMNOP'[i]}) {o}" for i, o in enumerate(options)]
    return f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


@torch.no_grad()
def decide(model, kind, batch):
    """batch: list of (state, question, options). Returns list of probability vectors."""
    T = [text(kind, *x) for x in batch]
    enc = tok(T, return_tensors="pt", padding=True, add_special_tokens=False, padding_side="right")
    h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    last = enc["attention_mask"].sum(1) - 1
    hl = h[torch.arange(len(T)), last].float()
    z = (hl @ W.T + b) / TEMP[kind]
    s0, _ = CFG["slots"]["ranges"][kind]
    return [torch.softmax(z[i, s0:s0 + len(batch[i][2])], 0).tolist() for i in range(len(T))]


if __name__ == "__main__":
    t = time.time(); model = load(); print("loaded", time.time() - t, flush=True)
    t = time.time()
    print(decide(model, "noul", [("Customer says the parcel arrived damaged and wants their money back.",
                                  "Is the customer asking for a refund?", ["false", "true"])]), time.time() - t)
