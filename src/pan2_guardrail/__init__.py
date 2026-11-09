"""PAN2 guardrail: Qwen3Guard on the prompt and Q16 on the generated video."""

from .safety_checker import PAN2SafetyChecker


__version__ = "2.0.0"

__all__ = ["PAN2SafetyChecker"]
