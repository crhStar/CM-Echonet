"""Evaluate CM-EchoNet on cached WavLM features.

Example:
  python evaluate.py --manifest data/manifests/iemocap.json \
      --cache data/cache_wavlm_features --session Session5
"""
import argparse
import json
from pathlib import Path
import torch
from model import WavLMEcho

LABELS = {0: 0, 4: 0, 1: 1, 2: 1, 3: 2, 5: 3}
NAMES = ["Angry", "Happy", "Sad", "Neutral"]


def map_labels(raw):
    return torch.tensor([LABELS.get(int(v), -100) for v in raw], dtype=torch.long)


def metrics(y_true, y_pred):
    out = []
    for k in range(4):
        tp = sum(t == k and p == k for t, p in zip(y_true, y_pred))
        support = sum(t == k for t in y_true)
        fp = sum(t != k and p == k for t, p in zip(y_true, y_pred))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(support, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        out.append((recall, f1, support))
    return {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1),
        "macro_f1": sum(x[1] for x in out) / 4,
        "uar": sum(x[0] for x in out) / 4,
        "per_class": {NAMES[i]: {"recall": r, "f1": f, "support": n} for i, (r, f, n) in enumerate(out)},
        "samples": len(y_true),
    }


def load_dialogue(path):
    item = torch.load(path, map_location="cpu", weights_only=False)
    features = item.get("features")
    labels = item.get("labels")
    if features is None or labels is None:
        raise ValueError(f"{path} must contain 'features' and 'labels'")
    return features.float(), map_labels(labels)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = WavLMEcho().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cache = Path(args.cache)
    selected = [x for x in manifest if args.session in (None, "all") or x.get("session") == args.session]
    if not selected:
        raise ValueError("No manifest entries match the requested session")
    true, pred = [], []
    with torch.inference_mode():
        for conv in selected:
            x, y = load_dialogue(cache / f"{conv['conversation_id']}.pt")
            logits, _ = model(x.to(device))
            p = logits.argmax(-1).cpu()
            mask = y >= 0
            true.extend(y[mask].tolist())
            pred.extend(p[mask].tolist())
    result = metrics(true, pred)
    print(json.dumps(result, indent=2))


def smoke_test(checkpoint):
    model = WavLMEcho().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    with torch.inference_mode():
        logits, shifts = model(torch.randn(7, 768))
    assert logits.shape == (7, 4) and shifts.shape == (7, 2)
    print(f"smoke test passed: logits={tuple(logits.shape)}, shift_logits={tuple(shifts.shape)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="weights/cm_echonet_iemocap_s5.pt")
    p.add_argument("--manifest")
    p.add_argument("--cache")
    p.add_argument("--session", default="Session5")
    p.add_argument("--device")
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()
    if a.smoke_test:
        smoke_test(a.checkpoint)
    elif not a.manifest or not a.cache:
        p.error("--manifest and --cache are required unless --smoke-test is used")
    else:
        main(a)
