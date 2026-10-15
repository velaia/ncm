"""Generate from your own text with a conditional checkpoint (WMT14 translation, XSum summarization).

    python scripts/prompt.py --config config/eval/wmt14.yaml "Guten Morgen, wie geht es dir?"
    echo "Ein Satz pro Zeile." | python scripts/prompt.py --config config/eval/wmt14.yaml
    python scripts/prompt.py --config config/eval/wmt14.yaml            # interactive prompt
    python scripts/prompt.py --config config/eval/wmt14.yaml --show-levels "Guten Morgen!"
"""
import argparse, os, sys
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer
from utils.misc import default_device

ap = argparse.ArgumentParser(description="generate from text input with a conditional NCM checkpoint")
ap.add_argument("text", nargs="*", help="input text(s); omit to read lines from stdin or prompt interactively")
ap.add_argument("--config", required=True, help="eval config, e.g. config/eval/wmt14.yaml")
ap.add_argument("--checkpoint-dir", default="", help="default artifacts/<task>")
ap.add_argument("--cfg", type=float, default=None, help="classifier-free guidance scale (paper uses 1.0 for WMT14)")
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--top_k", type=int, default=None)
ap.add_argument("--top_p", type=float, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show-levels", action="store_true", help="also decode the partial output after every level, coarse to fine")
a = ap.parse_args()

CFG = OmegaConf.load(a.config)
G = CFG.generation
P = int(CFG.model.prefix_len)
if P <= 1 or CFG.tokenizer.pre_tokenizer.name != "hf":
    sys.exit(f"{CFG.task} is not conditioned on text; use a text-to-text config such as wmt14 or xsum")
DEV = default_device()
ck = a.checkpoint_dir or os.path.join(ROOT, "artifacts", CFG.task)
pick = lambda cli, key, default: cli if cli is not None else G.get(key, default)
temp, top_k, top_p, cfg = pick(a.temp, "temperature", 1.0), pick(a.top_k, "top_k", 1), pick(a.top_p, "top_p", 0.0), pick(a.cfg, "cfg_scale", 0.0)

CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(DEV).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(DEV)
net.load_state_dict(torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)["model"], strict=True)
net.eval()
src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
vq = tk.vqvae.vq


def dec(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


def level_zhats(idx):
    # cumulative latent after each level (vq.idx_to_zhats accumulates in place, so its list aliases one tensor)
    z_hat = torch.zeros(idx.shape[0], vq.levels[-1], vq.embed_dim, device=idx.device)
    out, acc = [], 0
    for li, n in enumerate(vq.levels):
        q = vq.embeddings[li](idx[:, acc:acc + n])
        if li != len(vq.levels) - 1:
            q = vq._up_to_full(q, li, vq.levels[-1])
        z_hat = z_hat + vq._apply_quant_resi(q, li)
        out.append(z_hat)
        acc += n
    return out


@torch.no_grad()
def generate(text):
    ids = src_tok(text)["input_ids"]
    if len(ids) > P:
        print(f"  (input is {len(ids)} tokens; the model sees only the first {P})", file=sys.stderr)
    x = torch.tensor([(ids[:P] + [PAD] * P)[:P]], device=DEV)
    torch.manual_seed(a.seed)
    g = net.generate(None, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg, cfg_null_eos=EOS)
    if a.show_levels:
        for li, z in enumerate(level_zhats(g)):
            print(f"  L{li:<2} ({vq.levels[li]:>3} codes) {dec(tk.vqvae.decoder(z).argmax(-1)[0].tolist())}")
    return dec(tk.vqvae.decoder(level_zhats(g)[-1]).argmax(-1)[0].tolist())


print(f"[{CFG.task} on {DEV}, cfg={cfg}]", file=sys.stderr)
if a.text:
    for t in a.text:
        print(generate(t))
elif not sys.stdin.isatty():
    for line in sys.stdin:
        if line.strip():
            print(generate(line.strip()), flush=True)
else:
    print("enter text, empty line or Ctrl-D to quit", file=sys.stderr)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        print(generate(line))
