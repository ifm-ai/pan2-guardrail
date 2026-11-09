"""Convert Q16 for CLIP ViT-L/14 into a safetensors folder for the Hugging Face Hub.

The folder holds the OpenAI CLIP ViT-L/14 image encoder (`visual.*`, in the dtypes it runs with) and the two Q16
prompt embeddings (`embeddings`, rows `[appropriate, inappropriate]`), both unmodified:

    python scripts/convert_q16_to_safetensors.py \
        --clip_checkpoint ~/.cache/clip/ViT-L-14.pt \
        --q16_prompts Q16/data/ViT-L-14/prompts.p \
        --output_path q16-vit-l-14

`--clip_checkpoint` is the `ViT-L/14` checkpoint that `clip.load` downloads, and `--q16_prompts` is
`data/ViT-L-14/prompts.p` of https://github.com/ml-research/Q16 at commit 33de788. The prompts file is a pickle: only
load a copy you trust.
"""

import argparse
import json
import pickle
from pathlib import Path

import torch
from clip.model import VisionTransformer, convert_weights
from safetensors.torch import save_file


CONFIG = {"input_resolution": 224, "patch_size": 14, "width": 1024, "layers": 24, "heads": 16, "output_dim": 768}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip_checkpoint", required=True)
    parser.add_argument("--q16_prompts", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    state_dict = torch.jit.load(args.clip_checkpoint, map_location="cpu").state_dict()
    visual = VisionTransformer(**CONFIG)
    visual.load_state_dict({key[7:]: value for key, value in state_dict.items() if key.startswith("visual.")})
    convert_weights(visual)

    with open(args.q16_prompts, "rb") as f:
        embeddings = torch.from_numpy(pickle.load(f))
    if embeddings.shape != (2, CONFIG["output_dim"]):
        raise ValueError(f"Expected Q16 embeddings of shape (2, 768), got {tuple(embeddings.shape)}.")

    tensors = {f"visual.{key}": value.contiguous() for key, value in visual.state_dict().items()}
    tensors["embeddings"] = embeddings.contiguous()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path / "model.safetensors", metadata={"format": "pt"})
    (output_path / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")


if __name__ == "__main__":
    main()
