"""Inference from exported safetensors; training uses RAVEN's standard engine."""

from engines.diffusion_finetuning import DiffusionFinetuning


class InferenceEngine(DiffusionFinetuning):
    def run(self):
        # Initial weights are loaded before distributed placement. No optimizer or
        # training-data cursor is restored when evaluating a released checkpoint.
        self.validate(step=416)
