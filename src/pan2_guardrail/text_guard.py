"""Qwen3Guard-Gen input guard for the prompt."""

from __future__ import annotations

import re

import torch
from torch import nn


QWEN3GUARD_ID = "Qwen/Qwen3Guard-Gen-4B"
MAX_PROMPT_CHARS = 50_000


def parse_qwen3guard_output(output: str) -> tuple[str | None, str | None]:
    """The `Safety:` label (`Safe`, `Unsafe` or `Controversial`, `None` if missing) and the `Categories:` line."""
    label = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", output, re.IGNORECASE)
    categories = re.search(r"Categories:\s*(.+)", output)
    return (
        label.group(1).title() if label else None,
        categories.group(1).strip() if categories else None,
    )


class Qwen3GuardTextGuard(nn.Module):
    r"""
    Qwen3Guard-Gen classifying the prompt, in bfloat16 with greedy decoding.

    Args:
        model_path (`str`, defaults to `"Qwen/Qwen3Guard-Gen-4B"`): A Hub id or a local directory.
    """

    def __init__(self, model_path: str = QWEN3GUARD_ID):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).eval()

    @property
    def device(self) -> torch.device:
        return self.model.device

    @torch.no_grad()
    def forward(self, prompt: str) -> str:
        """Qwen3Guard's response, such as `"Safety: Unsafe\nCategories: Violent"`."""
        if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"The prompt must be non-empty and at most {MAX_PROMPT_CHARS:,} characters.")
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
