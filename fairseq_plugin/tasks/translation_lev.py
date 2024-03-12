# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    load_langpair_dataset,
)

from fairseq_plugin.utils import LinearScheduler

NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask", "bert_mask"])


@dataclass
class TranslationLevenshteinConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(default="random_delete", metadata={"help": "type of noise"})
    noise_schedule: Optional[str] = field(
        default=None, metadata={"help": "schedule for random mask. If it is not given, use uniform mask"}
    )


@register_task("translation_lev", dataclass=TranslationLevenshteinConfig)
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: TranslationLevenshteinConfig

    def __init__(self, cfg: TranslationLevenshteinConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.noise_scheduler = LinearScheduler(cfg.noise_schedule)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
            shuffle=(split != "test"),
        )

    def inject_noise(self, target_tokens, noise_ratio=None):
        pad = self.tgt_dict.pad()
        bos = self.tgt_dict.bos()
        eos = self.tgt_dict.eos()
        unk = self.tgt_dict.unk()

        def _random_delete(target_tokens):
            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2 + ((target_length - 2) * target_score.new_zeros(target_score.size(0), 1).uniform_()).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[:, : prev_target_tokens.ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_mask(target_tokens, noise_ratio):
            target_masks = target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            if noise_ratio is None:
                noise_ratio = target_length.clone().uniform_()

            target_length = target_length * noise_ratio
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = utils.new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            target_mask = target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        def _bert_mask(target_tokens):
            mask_prob = 0.15
            leave_unmasked_prob = 0.1
            random_replace_prob = 0.1

            target_masks = target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * mask_prob
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = utils.new_arange(target_rank) < target_length[:, None].long()
            mask = target_cutoff.scatter(1, target_rank, target_cutoff)

            rand_or_unmask_prob = random_replace_prob + leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (torch.rand_like(mask.float()) < rand_or_unmask_prob)
                if random_replace_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
                    decision = torch.rand_like(mask.float()) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            prev_target_tokens = target_tokens.clone()
            prev_target_tokens[mask] = unk

            if rand_mask is not None:
                prev_target_tokens[rand_mask] = torch.randint_like(
                    prev_target_tokens[rand_mask],
                    low=self.tgt_dict.nspecial,
                    high=len(self.tgt_dict),
                )

            return prev_target_tokens

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens, noise_ratio)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "bert_mask":
            return _bert_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq_plugin.iterative_refinement_generator import (
            IterativeRefinementGenerator,
        )

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
            len_penalty=getattr(args, "lenpen", 1.0),
            length_beam_topk=getattr(args, "length_beam_topk", False),
            decode_with_oracle_length=getattr(args, "decode_with_oracle_length", False),
            remove_consecutive_repetition=getattr(args, "remove_consecutive_repetition", False),
            merge_beam_results=getattr(args, "merge_beam_results", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError("Constrained decoding with the translation_lev task is not supported")

        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary, append_bos=True)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        noise_ratio = self.noise_scheduler.get_value(update_num)
        sample["prev_target"] = self.inject_noise(sample["target"], noise_ratio)

        loss, sample_size, logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        update_num = model.get_num_updates()
        noise_ratio = self.noise_scheduler.get_value(update_num)
        sample["prev_target"] = self.inject_noise(sample["target"], noise_ratio=noise_ratio)
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output
