# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq_plugin.iterative_refinement_generator import DecoderOut

from .fairseq_nat_model import (
    FairseqNATDecoder,
    FairseqNATEncoder,
    FairseqNATModel,
    ensemble_decoder,
    ensemble_encoder,
)


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = ((enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _softcopy_assignment(src_lens, trg_lens, tau=0.3):
    max_trg_len = trg_lens.max()
    max_src_len = src_lens.max()
    index_s = utils.new_arange(src_lens, max_src_len).float()
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    diff = (index_t[:, None] - index_s[None, :]).abs().neg()
    diff = diff.unsqueeze(0).expand(trg_lens.size(0), *diff.size())
    return diff / tau


def _uniform_assignment(src_lens, trg_lens, left_pad_src=True):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    # because the src tokens are left-padded, so if src tokens are not fully occupied,
    # copy index_t should skip the first few pad tokens instead of the last few non-pad tokens.
    if left_pad_src:
        index_t += (src_lens.max() - src_lens)[:, None]
    return index_t


@register_model("nonautoregressive_transformer")
class NATransformerModel(FairseqNATModel):
    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            default=None,
            choices=["uniform", "attention", "softcopy"],
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            "--pred-length-type",
            default="mean",
            choices=["mean", "bert", "none"],
            help="which type is used for predicting the length",
        )

    def nonspecial_mask(self, token):
        return token.ne(self.pad) & token.ne(self.eos) & token.ne(self.bos)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = NATransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

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
                "mask": tgt_tokens.ne(self.pad),
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
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        word_ins_out = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )
        _scores, _tokens = word_ins_out.max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append({"tokens": output_tokens.clone(), "scores": output_scores.clone()})

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, beam_size=1, topk=False, length_tgt=None):
        # length prediction
        if length_tgt is None:
            length_tgt = self.decoder.forward_length_prediction(
                self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out,
                beam_size=beam_size,
            )
            if beam_size > 1 and not topk:
                length_tgt = length_tgt[:, [0]] + utils.new_arange(length_tgt, 1, beam_size) - beam_size // 2
            length_tgt = length_tgt.view(-1)

        initial_output_tokens = self._get_initial_output_tokens(src_tokens, length_tgt)
        initial_output_scores = initial_output_tokens.new_zeros(*initial_output_tokens.size()).type_as(
            encoder_out["encoder_out"][0]
        )

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=1,
            max_step=1,
            history=None,
        )

    def _get_initial_output_tokens(self, src_tokens, length_tgt):
        length_tgt = length_tgt.clamp_(min=2, max=self.args.max_target_positions)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(length_tgt.size(0), max_length).fill_(self.pad)
        initial_output_tokens.masked_fill_(idx_length[None, :] < length_tgt[:, None], self.unk)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens

    @staticmethod
    def filter_terminated(decoder_out, not_terminated):
        return decoder_out._replace(
            output_tokens=decoder_out.output_tokens[not_terminated],
            output_scores=decoder_out.output_scores[not_terminated],
            attn=decoder_out.attn[not_terminated]
            if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
            else None,
            history=[{k: h[k][not_terminated] for k in h} for h in decoder_out.history]
            if decoder_out.history is not None
            else None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        utils.deprecation_warning("function regenerate_length_beam has been deprecated!")

        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = length_tgt[:, None] + utils.new_arange(length_tgt, 1, beam_size) - beam_size // 2
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(length_tgt.size(0), max_length).fill_(self.pad)
        initial_output_tokens.masked_fill_(idx_length[None, :] < length_tgt[:, None], self.unk)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(*initial_output_tokens.size()).type_as(
            decoder_out.output_scores
        )

        return decoder_out._replace(output_tokens=initial_output_tokens, output_scores=initial_output_scores)


class NATransformerEncoder(FairseqNATEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.length_token = nn.Embedding(1, embed_tokens.embedding_dim) if args.pred_length_type == "bert" else None

    @ensemble_encoder
    def forward(self, src_tokens, src_lengths=None, return_all_hiddens=False, token_embeddings=None):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        x, encoder_embedding, positions = self.forward_embedding(src_tokens, token_embeddings, return_pe=True)

        if self.length_token is not None:  # prepend a special length token embedding
            length_embed = self.length_token(src_tokens.new_zeros(src_tokens.size(0), 1))
            x = torch.cat([length_embed, x], dim=1)
            encoder_padding_mask = torch.cat(
                [encoder_padding_mask.new_zeros(src_tokens.size(0), 1), encoder_padding_mask], dim=1
            )

        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x if self.length_token is None else x[1:])

        # encoder layers
        for idx, layer in enumerate(self.layers):
            x, layer_attn, fc_result = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                need_head_weights=idx in self.need_attn_layers,
            )

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x if self.length_token is None else x[1:])
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x] if self.length_token is None else [x[1:], x[0]],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]  # B x T
            if self.length_token is None
            else [encoder_padding_mask[:, 1:]],
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
            "position_embedding": [positions],
        }


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", None)
        self.pred_length_type = getattr(args, "pred_length_type", "mean")
        if self.pred_length_type != "none":
            self.embed_length = Embedding(
                256 if self.pred_length_offset else args.max_target_positions,
                self.encoder_embed_dim,
                padding_idx=None,
            )
        else:
            self.embed_length = None

        if self.src_embedding_copy == "attention":
            self.attention_copy_linear = nn.Linear(self.embed_dim, self.embed_dim)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=1, return_all_hiddens=False, **kwargs):
        embedding_copy = False
        if self.src_embedding_copy and prev_output_tokens.eq(self.unk).any():
            embedding_copy = True

        features, inner_states = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=embedding_copy,
            return_all_hiddens=return_all_hiddens,
            **kwargs,
        )
        decoder_out = self.output_layer(features)

        if normalize:
            decoder_out = F.log_softmax(decoder_out, -1)
        if not return_all_hiddens:
            return decoder_out
        return decoder_out, inner_states

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        if self.pred_length_type == "mean":
            enc_feats = encoder_out["encoder_out"][0]  # T x B x C
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
            else:
                src_masks = None
            enc_feats = _mean_pooling(enc_feats, src_masks)
        elif self.pred_length_type == "bert":
            enc_feats = encoder_out["encoder_out"][1]

        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        return_all_hiddens=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy:
            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens, states=self.forward_copying_source(encoder_out, prev_output_tokens)
            )

        else:
            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attns = []
        inner_states = []
        if return_all_hiddens:
            inner_states.append(x)

        # decoder layers
        for i, layer in enumerate(self.layers):
            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0)
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                need_head_weights=i in self.need_attn_layers,
            )
            if return_all_hiddens:
                inner_states.append(x)
            attns.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attns": attns, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None, return_pe=False):
        # embed positions
        positions = self.embed_positions(prev_output_tokens) if self.embed_positions is not None else None

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if states is not None:
            unk_mask = prev_output_tokens.eq(self.unk)
            x[unk_mask] = states[unk_mask]

        if positions is not None:
            if self.concat_pos_linear is not None:
                x = self.concat_pos_linear(torch.cat([x, positions]), dim=-1)
            else:
                x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        if return_pe:
            return x, decoder_padding_mask, positions
        return x, decoder_padding_mask

    def forward_copying_source(self, encoder_out, prev_output_tokens):
        src_embed = encoder_out["encoder_embedding"][0]
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_mask = encoder_out["encoder_padding_mask"][0]
        else:
            src_mask = None
        src_mask = ~src_mask if src_mask is not None else prev_output_tokens.new_ones(*src_embed.size()[:2]).bool()
        tgt_mask = prev_output_tokens.ne(self.padding_idx)

        length_sources = src_mask.sum(1)
        length_targets = tgt_mask.sum(1)

        # TODO: currently embedding copy could assign BOS or EOS embedding to valid target tokens.
        if self.src_embedding_copy == "uniform":
            mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(~tgt_mask, 0)
            copied_embedding = torch.gather(
                src_embed,
                1,
                mapped_inputs.unsqueeze(-1).expand(*mapped_inputs.size(), src_embed.size(-1)),
            )

        elif self.src_embedding_copy == "attention":
            # use target position to attend positioned source embedding
            target_positions = self.embed_positions(prev_output_tokens)
            source_positions = encoder_out["position_embedding"][0]
            attn_scores = torch.bmm(
                self.attention_copy_linear(target_positions),
                (src_embed + source_positions).transpose(1, 2),
            )
            if src_mask is not None:
                attn_scores = attn_scores.masked_fill(~src_mask.unsqueeze(1), float("-inf"))
            copied_embedding = torch.bmm(attn_scores.softmax(-1), src_embed)

        elif self.src_embedding_copy == "softcopy":
            mapped_distance = _softcopy_assignment(
                length_sources,
                length_targets,
            ).type_as(src_embed)
            if src_mask is not None:
                mapped_distance = mapped_distance.masked_fill(~src_mask.unsqueeze(1), float("-inf"))
            copied_embedding = torch.bmm(mapped_distance.softmax(-1), src_embed)

        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None, beam_size=1):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(enc_feats.size(0))
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + self.embed_length.num_embeddings // 2
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=self.embed_length.num_embeddings - 1)

        else:
            pred_lengs = length_out.topk(beam_size, -1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - self.embed_length.num_embeddings // 2 + src_lengs[:, None]
            else:
                length_tgt = pred_lengs

        return length_tgt


@register_model_architecture("nonautoregressive_transformer", "nonautoregressive_transformer")
def base_architecture(args):
    from fairseq.models.transformer import (
        base_architecture as transformer_base_architecture,
    )

    transformer_base_architecture(args)


@register_model_architecture("nonautoregressive_transformer", "nonautoregressive_transformer_wmt_en_de")
def nat_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("nonautoregressive_transformer", "nonautoregressive_transformer_iwslt_de_en")
def nat_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
