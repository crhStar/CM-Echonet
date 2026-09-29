# CM-EchoNet Inference Package

This directory contains the inference-only release for the Causal Multi-Scale Emotion Echo Network (CM-EchoNet). It includes the model definition, the selected IEMOCAP Session 5 checkpoint, an evaluation script, and the minimum Python dependencies. The training code and raw IEMOCAP audio are intentionally excluded.

## What is included

```text
model.py                         CM-EchoNet architecture
evaluate.py                      checkpoint loading, inference, and metrics
weights/cm_echonet_iemocap_s5.pt Checkpoint for the reported 72.26/70.75/71.94 IEMOCAP run
requirements.txt                 minimum runtime dependencies
```

The checkpoint expects one cached dialogue file per conversation. Each file must be a PyTorch dictionary with:

```python
{
    "features": torch.Tensor,  # [number_of_turns, 768], cached WavLM features
    "labels": list[int],       # original IEMOCAP labels; -1 is ignored
    "session": "Session5"
}
```

The label mapping used by the paper is: Angry = {0, 4}, Happy = {1, 2}, Sad = {3}, Neutral = {5}.

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Verify the package without data

```bash
python evaluate.py --smoke-test
```

This checks that the checkpoint loads strictly into the released architecture and that a dialogue sequence produces four-class logits.

## Evaluate cached features

Place the paper's manifest and cached WavLM feature files in a local data directory, then run:

```bash
python evaluate.py \
  --manifest path/to/iemocap.json \
  --cache path/to/cache_wavlm_features \
  --session Session5
```

Use `--session all` to evaluate every manifest entry:

```bash
python evaluate.py --manifest path/to/iemocap.json \
  --cache path/to/cache_wavlm_features --session all
```

The script prints accuracy, macro-F1, UAR, sample count, and per-class recall/F1 as JSON. It uses only the observed chronological prefix of each dialogue and does not update model parameters.

## Rebuilding the feature cache

The repository release does not redistribute IEMOCAP or WavLM features. IEMOCAP must be obtained from its official license holder. To reproduce the cache, use WavLM `microsoft/wavlm-base-plus` and mean-pool its frame-level output for each utterance. The cache format above is the interface required by this package.

## Reproducibility note

The checkpoint corresponds to the IEMOCAP protocol in the paper: Sessions 1-3 for training, Session 4 for validation, and Session 5 for testing. The package is for evaluation only; no claim is made that the checkpoint is a universal or domain-invariant model.
