"""The PAN2 safety checker: Qwen3Guard on the prompt and Q16 on the generated video."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from torch import nn

from .text_guard import QWEN3GUARD_ID, Qwen3GuardTextGuard, parse_qwen3guard_output
from .video_guard import Q16_ID, Q16VideoGuard


logger = logging.getLogger(__name__)

NUM_VIDEO_FRAMES = 16


def sample_frame_indices(num_frames: int, num_samples: int = NUM_VIDEO_FRAMES) -> list[int]:
    """Uniformly spaced frame indices, first and last included."""
    if num_frames < 1:
        raise ValueError("Cannot check an empty video.")
    return np.unique(np.linspace(0, num_frames - 1, min(num_samples, num_frames)).round().astype(int)).tolist()


def load_config(
    pretrained_model_name_or_path: str | os.PathLike, subfolder: str = "", revision: str | None = None
) -> dict:
    """The `config.json` of a checker: the two guard paths, and optionally `threshold` and `min_frames`."""
    path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            str(pretrained_model_name_or_path), "config.json", subfolder=subfolder or None, revision=revision
        )
    with open(path) as f:
        return json.load(f)


class PAN2SafetyChecker(nn.Module):
    r"""
    Checks the prompt before generation and the generated video after it.

    The prompt is blocked unless Qwen3Guard labels it `Safe`. The video is blocked when at least `min_frames` of 16
    uniformly spaced frames reach `threshold` under Q16. Errors are raised, not caught: a caller that cannot complete
    a check must treat it as a block.

    Args:
        text_guard (`Qwen3GuardTextGuard`): The input guard.
        video_guard (`Q16VideoGuard`): The output guard.
        threshold (`float`, defaults to `0.8`): The Q16 score at which a frame is unsafe. Higher blocks fewer videos.
        min_frames (`int`, defaults to `1`): The number of unsafe frames that block a video.
    """

    def __init__(
        self,
        text_guard: Qwen3GuardTextGuard,
        video_guard: Q16VideoGuard,
        threshold: float = 0.8,
        min_frames: int = 1,
    ):
        super().__init__()
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"`threshold` must be in (0, 1), got {threshold}.")
        if min_frames < 1:
            raise ValueError(f"`min_frames` must be at least 1, got {min_frames}.")
        self.text_guard = text_guard
        self.video_guard = video_guard
        self.threshold = threshold
        self.min_frames = min_frames

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
        subfolder: str = "",
        revision: str | None = None,
        variant: str | None = None,
        dtype: torch.dtype | None = None,
        text_guard_path: str = QWEN3GUARD_ID,
        video_guard_path: str = Q16_ID,
        **kwargs,
    ) -> "PAN2SafetyChecker":
        """
        Load both guards on CPU from Hub ids or local directories.

        Without `pretrained_model_name_or_path`, the guards load from `text_guard_path` and `video_guard_path`. With
        it, they load from the guard paths in the `config.json` there (a Hub repo at `revision` or a local directory,
        in `subfolder`), which can also set `threshold` and `min_frames`. `variant` and `dtype` are accepted for
        diffusers' component loading and ignored: the guards keep their own dtypes. `kwargs` go to the constructor and
        take precedence over `config.json`.
        """
        if pretrained_model_name_or_path is not None:
            config = load_config(pretrained_model_name_or_path, subfolder, revision)
            text_guard_path = config.pop("text_guard")
            video_guard_path = config.pop("video_guard")
            kwargs = {**config, **kwargs}
        return cls(Qwen3GuardTextGuard(text_guard_path), Q16VideoGuard(video_guard_path), **kwargs)

    @property
    def device(self) -> torch.device:
        return self.video_guard.device

    @property
    def dtype(self) -> torch.dtype:
        # The dtype of the first parameter, as for a diffusers model, so that a diffusers pipeline can move the checker
        # with its other components; the guards keep their own dtypes.
        return next(self.parameters()).dtype

    def check_text_safety(self, prompt: str) -> bool:
        """Whether the prompt may be generated."""
        label, categories = parse_qwen3guard_output(self.text_guard(prompt))
        if label != "Safe":
            logger.critical("PAN2 guardrail blocked the prompt: Qwen3Guard label %s (%s).", label, categories)
            return False
        return True

    def check_video_safety(self, frames: np.ndarray) -> np.ndarray | None:
        """Returns the uint8 `[num_frames, height, width, 3]` `frames` when they pass, and `None` otherwise."""
        scores = self.video_guard(frames[sample_frame_indices(len(frames))])
        num_unsafe = int((scores >= self.threshold).sum())
        if num_unsafe >= self.min_frames:
            logger.critical(
                "PAN2 guardrail blocked the video: %d of %d frames reach the Q16 threshold %s (max score %.4f).",
                num_unsafe,
                len(scores),
                self.threshold,
                scores.max(),
            )
            return None
        return frames
