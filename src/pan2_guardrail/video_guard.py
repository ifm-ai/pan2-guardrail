"""Q16 output guard: the OpenAI CLIP ViT-L/14 image encoder and two learned prompt embeddings."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor


Q16_ID = "IFM/PAN2-guardrail-Q16"


class Q16VideoGuard(nn.Module):
    r"""
    Q16 ([Schramowski et al., 2022](https://arxiv.org/abs/2202.06675)) scores how likely each frame is inappropriate:
    `softmax(100 * cos(clip_image(frame), [e_appropriate, e_inappropriate]))[1]`.

    The OpenAI CLIP image encoder runs in float16 with float32 layer norms, as OpenAI CLIP and Q16 run it on GPU.

    Args:
        model_path (`str`, defaults to `"IFM/PAN2-guardrail-Q16"`): A Hub id or a local directory.
    """

    def __init__(self, model_path: str = Q16_ID):
        super().__init__()
        from clip.model import VisionTransformer, convert_weights

        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_path)
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)

        self.visual = VisionTransformer(**config)
        convert_weights(self.visual)
        self.register_buffer("embeddings", torch.empty(2, config["output_dim"], dtype=torch.float16))
        self.load_state_dict(load_file(os.path.join(model_path, "model.safetensors")))
        self.eval()

        resolution = config["input_resolution"]
        self.preprocess = Compose(
            [
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(resolution),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    @property
    def device(self) -> torch.device:
        return self.embeddings.device

    @torch.no_grad()
    def forward(self, frames: np.ndarray) -> torch.Tensor:
        """The probability that each uint8 `[height, width, 3]` frame is inappropriate, as a float32 CPU tensor."""
        pixel_values = torch.stack([self.preprocess(Image.fromarray(frame).convert("RGB")) for frame in frames])
        features = self.visual(pixel_values.to(self.device, self.visual.conv1.weight.dtype)).float()
        logits = 100.0 * F.normalize(features, dim=-1) @ F.normalize(self.embeddings.float(), dim=-1).T
        return logits.softmax(dim=-1)[:, 1].cpu()
