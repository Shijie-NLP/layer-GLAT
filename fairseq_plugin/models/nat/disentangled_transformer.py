from collections import namedtuple
from typing import Any, Dict, List, Optional

import torch
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerConfig
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from torch import Tensor

from fairseq_plugin.iterative_refinement_generator import DecoderOut

from .nonautoregressive_transformer import (
    NATransformerDecoder,
    NATransformerModel,
    base_architecture,
)

DiscoOut = namedtuple("DiscoOut", DecoderOut._fields + ("self_attn_mask",))


def _additive_self_attn_mask(mask):
    if mask.dtype == torch.bool:
        return torch.masked_fill(mask.float(), mask, float("-inf"))
    return mask


def eyes_like(x: torch.Tensor) -> torch.Tensor:
    return torch.eye(*x.shape[-2:]).expand_as(x).to(x)


@register_model("disentangled_transformer")
class DisentangledTransformerModel(NATransformerModel):
    """
    See `"Non-autoregressive Machine Translation with Disentangled Context Transformer"
    (Kasai et al., 2020) <https://arxiv.org/abs/2001.05136>`_.
    """

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

        parser.add_argument(
            "--renew-rank",
            action="store_true",
            help="use renew rank for every iteration during inference",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DisentangledTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        self_attn_mask = self.decoder._build_easy_first_mask(tgt_tokens)

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            encoder_out=encoder_out,
            prev_output_tokens=tgt_tokens,  # use the ground-truth token
            self_attn_mask=self_attn_mask,
        )

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": self.nonspecial_mask(tgt_tokens),
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

        return ret

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = self.nonspecial_mask(output_tokens)
        if decoder_out.self_attn_mask is None:
            assert step == 1
            decoder_out = decoder_out._replace(
                self_attn_mask=output_masks.unsqueeze(1).repeat(1, output_tokens.size(1), 1)
            )

        word_ins_out = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
            self_attn_mask=decoder_out.self_attn_mask,
        )
        _scores, _tokens = word_ins_out.max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append({"tokens": output_tokens.clone(), "scores": output_scores.clone()})

        if step == 1 or getattr(self.args, "renew_rank", False):
            decoder_out = decoder_out._replace(
                self_attn_mask=self.decoder._build_easy_first_mask(output_tokens, output_scores)
            )

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, beam_size=1, topk=False, length_tgt=None):
        decoder_out = super().initialize_output_tokens(encoder_out, src_tokens, beam_size, topk, length_tgt)
        decoder_out = decoder_out._asdict()
        decoder_out["self_attn_mask"] = None
        return DiscoOut(**decoder_out)

    def filter_terminated(self, decoder_out, not_terminated):
        decoder_out = super().filter_terminated(decoder_out, not_terminated)
        return decoder_out._replace(
            self_attn_mask=decoder_out.self_attn_mask[not_terminated]
            if decoder_out.self_attn_mask is not None
            else None,
        )


class DisentangledTransformerDecoder(NATransformerDecoder):
    def _build_easy_first_mask(self, tokens, scores=None):
        pad_mask = tokens.eq(self.pad)
        bos_eos_mask = tokens.eq(self.bos) | tokens.eq(self.eos)

        if scores is None:
            bsz, seq_len = tokens.size()
            # each token has its own conditions, and always attend to eos and bos
            # a high score value means NOT attend and will be replaced by -inf
            rand_scores = torch.rand([bsz, seq_len, seq_len], device=tokens.device)
            rand_cutoff = torch.rand([bsz, seq_len], device=tokens.device)

            rand_scores.masked_fill_(pad_mask.unsqueeze(1), 2)
            rand_scores.masked_fill_(eyes_like(rand_scores).bool(), 2)
            rand_scores.masked_fill_(bos_eos_mask.unsqueeze(1), -2)

            sorted_tokens = rand_scores.sort(2)[1]
            ranks = sorted_tokens.sort(2)[1]

            # hardest case: only attend bos and eos, len=2
            # easiest case: attend to all but itself, len=seq_len - 1
            # so the range is [2, seq_len - 1]
            cutoff_len = seq_len - pad_mask.sum(-1, keepdim=True) - 2
            cutoff_len = (rand_cutoff * cutoff_len).long() + 2

            return ranks >= cutoff_len.unsqueeze(2)

        else:
            # only attend to tokens whose score is higher (easier)
            # we sort it in descending order, therefore the
            # lower the rank the higher the score
            # we do not attend to tokens whose rank is higher.
            sorted_tokens = scores.sort(1, descending=True)[1]
            ranks = sorted_tokens.sort(1)[1]

            mask = ranks.unsqueeze(1) >= ranks.unsqueeze(2)

            mask.masked_fill_(bos_eos_mask.unsqueeze(1), False)
            mask.masked_fill_(pad_mask.unsqueeze(1), True)
            return mask

    def build_decoder_layer(self, args, no_encoder_attn=False):
        cfg = TransformerConfig.from_namespace(args)
        layer = DiscoTransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        self_attn_mask=None,
        **unused,
    ):
        if embedding_copy:
            kv, decoder_padding_mask, positions = self.forward_embedding(
                prev_output_tokens,
                states=self.forward_copying_source(encoder_out, prev_output_tokens),
                return_pe=True,
            )
        else:
            kv, decoder_padding_mask, positions = self.forward_embedding(prev_output_tokens, return_pe=True)

        # B x T x C -> T x B x C
        kv = kv.transpose(0, 1)

        x = self.dropout_module(positions)
        x = x.transpose(0, 1)
        attns = []
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):
            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                kv=kv,
                encoder_out=encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_padding_mask=encoder_out["encoder_padding_mask"][0]
                if (encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0)
                else None,
                self_attn_mask=_additive_self_attn_mask(self_attn_mask),
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)
            attns.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attns": attns, "inner_states": inner_states}


class DiscoTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    def forward(
        self,
        x,
        kv=None,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[Tensor]] = None,
        prev_attn_state: Optional[List[Tensor]] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        self_attn_configs: Optional[Dict[str, Any]] = None,
        cross_attn_configs: Optional[Dict[str, Any]] = None,
    ):
        if need_head_weights:
            need_attn = True

        attns = {}

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat((x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(encoder_out.size(1), encoder_out.size(0))
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            assert encoder_out is not None
            y = torch.cat((encoder_out, x if kv is None else kv), dim=0)
        else:
            y = x if kv is None else kv

        self_attn_configs = self_attn_configs or {}

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
            static_kv=True,
            **self_attn_configs,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        attns["self_attn"] = attn

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            cross_attn_configs = cross_attn_configs or {}

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                **cross_attn_configs,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            attns["cross_attn"] = attn

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attns, self_attn_state
        return x, attns, None


@register_model_architecture("disentangled_transformer", "disentangled_transformer")
def disentangled_transformer_base_architecture(args):
    base_architecture(args)


@register_model_architecture("disentangled_transformer", "disentangled_transformer_wmt_en_de")
def disentangled_transformer_wmt_en_de(args):
    disentangled_transformer_base_architecture(args)


@register_model_architecture("disentangled_transformer", "disentangled_transformer_iwslt_de_en")
def disentangled_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    disentangled_transformer_base_architecture(args)
