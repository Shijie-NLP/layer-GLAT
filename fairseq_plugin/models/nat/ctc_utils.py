import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils

logger = logging.getLogger(__name__)


class CTCLoss(nn.Module):
    def __init__(self, blank_idx, label_smoothing=0.0, imputered_ctc=False):
        super().__init__()
        self.blank_idx = blank_idx
        self.label_smoothing = label_smoothing
        self.imputered_ctc = imputered_ctc

        try:
            from torch_imputer import best_alignment

            self._best_alignment = best_alignment

            if self.imputered_ctc:
                from torch_imputer import imputer_loss

                self.imputer_loss = imputer_loss

        except ImportError:
            raise RuntimeError(
                "Package torch_imputer is not found, cannot train CTC with Imputer loss."
                "Please use fairseq's CTC loss instead."
            )

    def forward(self, logits, tgt_tokens, input_mask, target_mask):
        def mean_ds(x: torch.Tensor, dim=None) -> torch.Tensor:
            return x.float().mean().type_as(x) if dim is None else x.float().mean(dim).type_as(x)

        input_lengths = input_mask.sum(-1)
        target_lengths = target_mask.sum(-1)

        lprobs = F.log_softmax(logits, dim=-1)
        losses = F.ctc_loss(
            lprobs.transpose(0, 1).float(),
            targets=tgt_tokens,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.blank_idx,
            reduction="none",
            zero_infinity=True,
        )

        ctc_loss = losses.sum() / input_mask.sum()
        if self.label_smoothing > 0:
            loss = ctc_loss * (1 - self.label_smoothing) - mean_ds(lprobs[input_mask]) * self.label_smoothing
        else:
            loss = ctc_loss

        # The number of invalid samples are samples where the predictions are shorter than the targets.
        # If this is true for too many samples, one might think about increasing --ctc-upsample-scale.
        n_invalid_samples = utils.item((input_lengths < target_lengths).sum())

        if n_invalid_samples > 0:
            logger.warning(
                f"{n_invalid_samples} samples have input length shorter than target length."
                f"Try to increase the factor `--glat-upsample-scale`"
            )

        return loss, ctc_loss

    def best_alignment(self, logits, target_tokens, input_mask, target_mask):
        lprobs = F.log_softmax(logits, dim=-1)
        best_aligns = self._best_alignment(
            lprobs.transpose(0, 1).float().contiguous(),
            targets=target_tokens,
            input_lengths=input_mask.sum(-1),
            target_lengths=target_mask.sum(-1),
            blank=self.blank_idx,
            zero_infinity=True,
        )

        pad_to_max_len = lambda a: a + [a[-1]] * (input_mask.size(1) - len(a))

        best_aligns_pad = lprobs.new_tensor([pad_to_max_len(a) for a in best_aligns], dtype=torch.long)
        best_aligned_pos = (best_aligns_pad // 2).clip(max=target_tokens.size(1) - 1)
        best_aligned_token = target_tokens.gather(-1, best_aligned_pos)
        # the odd positions should align to the blank position
        best_aligned_token = best_aligned_token.masked_fill(best_aligns_pad % 2 == 0, self.blank_idx)
        return best_aligned_token

    def ctc_inverse_mask(self, states, batch_first=True):
        if batch_first:
            states = states.transpose(0, 1)

        ctc_masks = torch.zeros_like(states, dtype=torch.bool)

        for t in range(states.shape[0] - 1):
            ctc_masks[t] = (states[t] == states[t + 1]) | (states[t] == self.blank_idx)

        ctc_masks[-1] = states[-1] == self.blank_idx

        return ctc_masks.transpose(0, 1) if batch_first else ctc_masks


def longest_common_subsequence(x, y, pad):
    dummy = x.new_empty(0)

    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()

    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    assert x.ndim == y.ndim
    assert x.shape[0] == y.shape[0]

    if pad is None:
        x_mask = np.ones_like(x)
        y_mask = np.ones_like(y)
    else:
        x_mask = x != pad
        y_mask = y != pad

    x_lens = x_mask.sum(-1)
    y_lens = y_mask.sum(-1)

    sequences = np.full_like(x if x.shape[1] < y.shape[1] else y, pad or 0)
    for bsz, (xl, yl) in enumerate(zip(x_lens, y_lens)):
        dp = [[0] * (yl + 1) for _ in range(xl + 1)]
        for i in range(1, xl + 1):
            for j in range(1, yl + 1):
                match = 1 if x[bsz, i - 1] == y[bsz, j - 1] else 0
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + match)

        cidx = sequences.shape[1] - 1
        while xl > 0 and yl > 0:
            match = 1 if x[bsz, xl - 1] == y[bsz, yl - 1] else 0

            if dp[xl][yl] == dp[xl - 1][yl - 1] + match:
                if match == 1:
                    sequences[bsz, cidx] = x[bsz, xl - 1]
                    cidx -= 1
                xl -= 1
                yl -= 1

            elif dp[xl][yl] == dp[xl - 1][yl]:
                xl -= 1

            else:
                yl -= 1

    lcs = torch.from_numpy(sequences).to(dummy)
    return utils.convert_padding_direction(lcs, pad, left_to_right=True)
