# PAN2 guardrail

The safety checker for [PAN2](https://huggingface.co/IFM/PAN2) video generation. It checks the prompt before
generation and the generated video after it:

- **Input guard:** [Qwen3Guard-Gen-4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B) classifies the prompt. Only
  `Safe` passes; `Unsafe`, `Controversial` and unparseable responses block it.
- **Output guard:** [Q16](https://github.com/ml-research/Q16) scores 16 uniformly spaced frames of the video with the
  OpenAI CLIP ViT-L/14 image encoder and two learned prompt embeddings. The video is blocked when at least
  `min_frames` (default 1) frames score at least `threshold` (default 0.8).

## Install

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail
```

## Usage

```python
import numpy as np
from pan2_guardrail import PAN2SafetyChecker

checker = PAN2SafetyChecker.from_pretrained().to("cuda")

checker.check_text_safety("A red fox trots through fresh snow.")  # False blocks the request
video = np.zeros((241, 704, 1248, 3), dtype=np.uint8)              # decoded uint8 frames [T, H, W, C]
checker.check_video_safety(video)                                  # the frames when they pass, None when blocked
```

A blocked check logs the Qwen3Guard label or the Q16 scores at `CRITICAL`. Errors are raised, not caught: a caller
that cannot complete a check must treat it as a block.

`from_pretrained` loads both guards on CPU from the Hub, or from local directories
(`text_guard_path`, `video_guard_path`); `threshold` and `min_frames` go to the constructor. Move the checker with
`.to(device)`, not `.to(dtype)`: Qwen3Guard runs in bfloat16 and the CLIP encoder in float16, about 8 GB and 0.6 GB.

## Models

| Guard | Model | Weights | License |
|---|---|---|---|
| Input | Qwen3Guard-Gen-4B | [`Qwen/Qwen3Guard-Gen-4B`](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B) | Apache-2.0 |
| Output | Q16 with OpenAI CLIP ViT-L/14 | [`IFM/PAN2-guardrail-Q16`](https://huggingface.co/IFM/PAN2-guardrail-Q16) | CLIP: MIT; Q16: see the model card |

`scripts/convert_q16_to_safetensors.py` converts the original OpenAI CLIP checkpoint and Q16 prompt embeddings into
the safetensors folder on the Hub.

## License and Attribution

MBZUAI's own code in this repository is released under the [Apache License, Version 2.0](LICENSE).

This repository also includes third-party components, each under its own license. The license files are:

- [LICENSE](LICENSE): the license for MBZUAI's own code in this repository.
- [NOTICE](NOTICE): MBZUAI's copyright notice and attribution for the third-party components.
- [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES): the full license text of each third-party component, with its original copyright line.

### Modifications

The Qwen3Guard-Gen-4B files and weights are used as released by their authors and have not been modified or fine-tuned by MBZUAI. PAN 2 uses Qwen3Guard-Gen-4B and Q16 for content moderation. Q16's pre-trained weights (the learned prompt parameters applied to CLIP) are used as released by its authors, who trained them on the SMID dataset; the decision threshold applied to Q16's outputs has been calibrated by MBZUAI.

## Citation

```bibtex
@article{zhao2025qwen3guard,
  title   = {Qwen3Guard Technical Report},
  author  = {Zhao, Haiquan and Yuan, Chenhan and Huang, Fei and Hu, Xiaomeng and Zhang, Yichang and Yang, An and Yu, Bowen and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang and others},
  journal = {arXiv preprint arXiv:2510.14276},
  year    = {2025}
}

@inproceedings{schramowski2022q16,
  title     = {Can Machines Help Us Answering Question 16 in Datasheets, and In Turn Reflecting on Inappropriate Content?},
  author    = {Schramowski, Patrick and Tauchmann, Christopher and Kersting, Kristian},
  booktitle = {ACM Conference on Fairness, Accountability, and Transparency (FAccT)},
  year      = {2022}
}
```
