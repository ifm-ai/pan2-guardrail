import json
import os

import numpy as np
import pytest
import torch
from torch import nn

from pan2_guardrail import PAN2SafetyChecker
from pan2_guardrail import safety_checker
from pan2_guardrail.safety_checker import sample_frame_indices
from pan2_guardrail.text_guard import parse_qwen3guard_output


class FakeTextGuard(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, prompt):
        return self.output


class FakeVideoGuard(nn.Module):
    """Scores each frame with its first pixel / 100, so a test sets the scores through the frames."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.checked = None

    @property
    def device(self):
        return self.weight.device

    def forward(self, frames):
        self.checked = frames
        return torch.tensor([frame[0, 0, 0] / 100 for frame in frames], dtype=torch.float32)


def make_checker(text_output="Safety: Safe\nCategories: None", **kwargs):
    return PAN2SafetyChecker(FakeTextGuard(text_output), FakeVideoGuard(), **kwargs)


def make_video(scores, num_frames=313):
    video = np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)
    for index, score in zip(sample_frame_indices(num_frames), scores):
        video[index, 0, 0, 0] = round(score * 100)
    return video


def test_sample_frame_indices():
    assert sample_frame_indices(241) == list(range(0, 241, 16))
    assert sample_frame_indices(313) == [0, 21, 42, 62, 83, 104, 125, 146, 166, 187, 208, 229, 250, 270, 291, 312]
    assert sample_frame_indices(5) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        sample_frame_indices(0)


def test_parse_qwen3guard_output():
    assert parse_qwen3guard_output("Safety: Safe\nCategories: None") == ("Safe", "None")
    assert parse_qwen3guard_output("Safety: Unsafe\nCategories: Violent") == ("Unsafe", "Violent")
    assert parse_qwen3guard_output("no label") == (None, None)


@pytest.mark.parametrize(
    "output, safe",
    [
        ("Safety: Safe\nCategories: None", True),
        ("Safety: Unsafe\nCategories: Violent", False),
        ("Safety: Controversial\nCategories: Politically Sensitive Topics", False),
        ("unparseable", False),
    ],
)
def test_check_text_safety(output, safe):
    assert make_checker(output).check_text_safety("a cat") is safe


def test_check_video_safety_checks_16_frames():
    checker = make_checker()
    video = make_video([0.1] * 16)
    assert checker.check_video_safety(video) is video
    assert len(checker.video_guard.checked) == 16


@pytest.mark.parametrize(
    "scores, kwargs, safe",
    [
        ([0.1] * 15 + [0.8], {}, False),
        ([0.1] * 15 + [0.79], {}, True),
        ([0.1] * 15 + [0.85], {"threshold": 0.9}, True),
        ([0.1] * 15 + [0.9], {"min_frames": 2}, True),
        ([0.1] * 14 + [0.9, 0.9], {"min_frames": 2}, False),
    ],
)
def test_check_video_safety_threshold(scores, kwargs, safe):
    video = make_video(scores)
    assert (make_checker(**kwargs).check_video_safety(video) is video) is safe


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make_checker(threshold=1.0)
    with pytest.raises(ValueError):
        make_checker(min_frames=0)


@pytest.mark.skipif(not os.getenv("PAN2_GUARDRAIL_RUN_SLOW"), reason="downloads ~9 GB of weights")
def test_real_weights():
    checker = PAN2SafetyChecker.from_pretrained().to("cuda" if torch.cuda.is_available() else "cpu")
    assert checker.check_text_safety("A red fox trots through fresh snow in a pine forest at dawn.")
    video = np.full((241, 64, 64, 3), 128, dtype=np.uint8)
    assert checker.check_video_safety(video) is video


def test_from_pretrained_reads_config(tmp_path, monkeypatch):
    (tmp_path / "safety_checker").mkdir()
    (tmp_path / "safety_checker" / "config.json").write_text(
        json.dumps({"text_guard": "org/text-guard", "video_guard": "org/video-guard", "threshold": 0.9})
    )
    paths = {}
    monkeypatch.setattr(
        safety_checker, "Qwen3GuardTextGuard", lambda path: paths.setdefault("text", path) and FakeTextGuard("")
    )
    monkeypatch.setattr(
        safety_checker, "Q16VideoGuard", lambda path: paths.setdefault("video", path) and FakeVideoGuard()
    )

    checker = PAN2SafetyChecker.from_pretrained(
        tmp_path, subfolder="safety_checker", variant=None, dtype=torch.float16
    )
    assert paths == {"text": "org/text-guard", "video": "org/video-guard"}
    assert (checker.threshold, checker.min_frames) == (0.9, 1)
    assert checker.dtype == torch.float32
