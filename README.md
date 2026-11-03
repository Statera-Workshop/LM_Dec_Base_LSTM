# LM_Dec_Base_LSTM

LSTM - Base family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: (new) Legacy baseline, PyTorch nn.LSTM
- Upstream / license: torch.nn.LSTM (cuDNN); classic baseline requested 2026-10-22

## Differences from the source ([Unified] protocol)

- 4 layers, hidden 768, input dim 512, LayerNorm before head

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 18.44 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.651 -> 0.002 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.588 / test 1.595 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.487 / test 1.508 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 201 MB, inference 87 MB
- Peak GPU memory at length 8192 (batch 1): train 477 MB, inference 127 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.02 (keys removed 0.02); 4 pairs @ 2048: 0.02 (keys removed 0.03); 16 pairs @ 512: 0.02 (keys removed 0.01); 16 pairs @ 2048: 0.02 (keys removed 0.02)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.30 (keys removed 0.24); 4 pairs @ 2048: 0.24 (keys removed 0.22); 4 pairs @ 8192: 0.26 (keys removed 0.25); 16 pairs @ 512: 0.01 (keys removed 0.02); 16 pairs @ 2048: 0.01 (keys removed 0.02); 16 pairs @ 8192: 0.02 (keys removed 0.01)
- LM-06 memory update `lm06`: @512: latest 0.01, unchanged 0.01, stale 0.02; @2048: latest 0.02, unchanged 0.01, stale 0.01

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check lstm
python run.py lm02 lstm      # enwik8 is downloaded to data/ on first use
```
