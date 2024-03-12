# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""


import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.utils import model_eval, new_arange

from .nonautoregressive_transformer import NATransformerModel, base_architecture


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (output_masks.sum(1, keepdim=True).type_as(output_scores) * p).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("cmlm_transformer")
class CMLMNATransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.correct_ratio = getattr(args, "self_correct_ratio", 0)
        assert 0 <= self.correct_ratio <= 1

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

        parser.add_argument(
            "--self-correct-ratio",
            type=float,
            default=0.0,
            help="percentage of ground truth tokens replaced by the prediction",
        )
        parser.add_argument(
            "--self-correct-random-rank",
            action="store_true",
            help="use random rank, otherwise use least confident positions",
        )
        parser.add_argument(
            "--predict-all",
            action="store_true",
            help="predict all tokens, otherwise only predict the replaced ones",
        )

    def forward_self_correct(self, encoder_out, prev_output_tokens):
        nonspecial_mask = self.nonspecial_mask(prev_output_tokens)
        observed_token_mask = nonspecial_mask & prev_output_tokens.ne(self.unk)
        full_mask_tokens = prev_output_tokens.masked_fill(nonspecial_mask, self.unk)

        with torch.no_grad():
            # the first pass use full-masked input
            with model_eval(self):
                init_pass = self.decoder(normalize=True, prev_output_tokens=full_mask_tokens, encoder_out=encoder_out)
            init_scores, init_tokens = init_pass.max(-1)

        if self.correct_ratio == 1:
            # all observed tokens will be replaced with initial predictions
            replace_mask = observed_token_mask
        else:
            if getattr(self.args, "self_correct_random_rank", False):
                init_scores.uniform_()

            # only rank observed tokens
            init_scores.masked_fill_(~observed_token_mask, 2.0)
            # TODO: make sure _skeptical_unmasking works when bos and eos of init_scores is not 0
            replace_mask = _skeptical_unmasking(init_scores, observed_token_mask, self.correct_ratio)

        # tokens at True positions will be replaced by predictions
        prev_output_tokens = torch.where(replace_mask, init_tokens, prev_output_tokens)

        if self.correct_ratio == 1 or getattr(self.args, "predict_all", False):
            word_ins_mask = nonspecial_mask
        else:
            word_ins_mask = prev_output_tokens.eq(self.unk) | replace_mask

        return prev_output_tokens, word_ins_mask

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        if self.correct_ratio > 0:
            prev_output_tokens, word_ins_mask = self.forward_self_correct(encoder_out, prev_output_tokens)
        else:
            word_ins_mask = prev_output_tokens.eq(self.unk)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
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

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        if self.correct_ratio > 0:
            # if used self_correct, we always predict all tokens
            output_masks = self.nonspecial_mask(output_tokens)
        else:
            output_masks = output_tokens.eq(self.unk)

        word_ins_out = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        )
        _scores, _tokens = word_ins_out.max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append({"tokens": output_tokens.clone(), "scores": output_scores.clone()})

        # skeptical decoding (depend on the maximum decoding steps.)
        if step < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, self.nonspecial_mask(output_tokens), 1 - step / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append({"tokens": output_tokens.clone()})

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


@register_model_architecture("cmlm_transformer", "cmlm_transformer")
def cmlm_base_architecture(args):
    base_architecture(args)


@register_model_architecture("cmlm_transformer", "cmlm_transformer_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)


@register_model_architecture("cmlm_transformer", "cmlm_transformer_iwslt_de_en")
def cmlm_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    cmlm_base_architecture(args)
