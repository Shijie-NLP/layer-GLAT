# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.models import register_model, register_model_architecture

from .nonautoregressive_transformer import NATransformerModel, base_architecture


def _sequential_poisoning(s, dict, beta=0.33, bos=2, eos=3, pad=1):
    # s: input batch
    # V: vocabulary size
    rand_words = torch.randint(low=dict.nspecial, high=len(dict), size=s.size(), device=s.device)
    choices = torch.rand(size=s.size(), device=s.device)
    choices.masked_fill_((s == pad) | (s == bos) | (s == eos), 1)

    replace = choices < beta / 3
    repeat = (choices >= beta / 3) & (choices < beta * 2 / 3)
    swap = (choices >= beta * 2 / 3) & (choices < beta)
    safe = choices >= beta

    for i in range(s.size(1) - 1):
        rand_word = rand_words[:, i]
        next_word = s[:, i + 1]
        self_word = s[:, i]

        replace_i = replace[:, i]
        swap_i = swap[:, i] & (next_word != eos)
        repeat_i = repeat[:, i] & (next_word != eos)
        safe_i = safe[:, i] | ((next_word == eos) & (~replace_i))

        s[:, i] = self_word * (safe_i | repeat_i).long() + next_word * swap_i.long() + rand_word * replace_i.long()
        s[:, i + 1] = next_word * (safe_i | replace_i).long() + self_word * (swap_i | repeat_i).long()
    return s


def gumbel_noise(input, TINY=1e-8):
    return input.new_zeros(*input.size()).uniform_().add_(TINY).log_().neg_().add_(TINY).log_().neg_()


@register_model("iterative_nonautoregressive_transformer")
class IterNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.train_step = getattr(args, "train_step", 4)
        self.dae_ratio = getattr(args, "dae_ratio", 0.5)
        self.stochastic_approx = getattr(args, "stochastic_approx", False)

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        parser.add_argument(
            "--train-step",
            type=int,
            help="number of refinement iterations during training",
        )
        parser.add_argument(
            "--dae-ratio",
            type=float,
            help="the probability of switching to the denoising auto-encoder loss",
        )
        parser.add_argument(
            "--stochastic-approx",
            action="store_true",
            help="sampling from the decoder as the inputs for next iteration",
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        B, T = prev_output_tokens.size()

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # decoding
        word_ins_outs, word_ins_tgts, word_ins_masks = [], [], []
        for t in range(1, self.train_step + 1):
            word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
                step=t,
            )
            word_ins_tgt = tgt_tokens
            word_ins_mask = word_ins_tgt.ne(self.pad)

            word_ins_outs.append(word_ins_out)
            word_ins_tgts.append(word_ins_tgt)
            word_ins_masks.append(word_ins_mask)

            if t < self.train_step:
                # prediction for next iteration
                if self.stochastic_approx:
                    word_ins_prediction = (word_ins_out + gumbel_noise(word_ins_out)).max(-1)[1]
                else:
                    word_ins_prediction = word_ins_out.max(-1)[1]

                prev_output_tokens = prev_output_tokens.masked_scatter(
                    word_ins_mask, word_ins_prediction[word_ins_mask]
                )

                if self.dae_ratio > 0:
                    # we do not perform denoising for the first iteration
                    corrputed = torch.rand(size=(B,), device=prev_output_tokens.device) < self.dae_ratio
                    corrputed_tokens = _sequential_poisoning(
                        tgt_tokens[corrputed],
                        self.tgt_dict,
                        0.33,
                        self.bos,
                        self.eos,
                        self.pad,
                    )
                    prev_output_tokens[corrputed] = corrputed_tokens

        # concat everything
        word_ins_out = torch.cat(word_ins_outs, 0)
        word_ins_tgt = torch.cat(word_ins_tgts, 0)
        word_ins_mask = torch.cat(word_ins_masks, 0)

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": word_ins_tgt,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "report_accuracy": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }


@register_model_architecture("iterative_nonautoregressive_transformer", "iterative_nonautoregressive_transformer")
def inat_base_architecture(args):
    base_architecture(args)


@register_model_architecture(
    "iterative_nonautoregressive_transformer",
    "iterative_nonautoregressive_transformer_wmt_en_de",
)
def iter_nat_wmt_en_de(args):
    inat_base_architecture(args)


@register_model_architecture(
    "iterative_nonautoregressive_transformer",
    "iterative_nonautoregressive_transformer_iwslt_de_en",
)
def nat_iterative_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    inat_base_architecture(args)
