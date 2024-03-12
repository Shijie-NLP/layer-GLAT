from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from fairseq import meters, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerConfig
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq_plugin.utils import LinearScheduler

from .disentangled_transformer import DiscoTransformerDecoderLayerBase
from .nonautoregressive_transformer import (
    NATransformerDecoder,
    NATransformerModel,
    base_architecture,
    ensemble_decoder,
)


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (output_masks.sum(1, keepdim=True).type_as(output_scores) * p).long()
    skeptical_mask = utils.new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def _additive_self_attn_mask(mask):
    if mask.dtype == torch.bool:
        return torch.masked_fill(mask.float(), mask, float("-inf"))
    return mask


def layer_single_forward(layer, x, kv, encoder_out, self_attn_mask, decoder_padding_mask):
    return layer(
        x,
        kv=kv,
        encoder_out=encoder_out["encoder_out"][0]
        if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
        else None,
        encoder_padding_mask=encoder_out["encoder_padding_mask"][0]
        if (encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0)
        else None,
        self_attn_mask=_additive_self_attn_mask(self_attn_mask) if self_attn_mask is not None else None,
        self_attn_padding_mask=decoder_padding_mask,
    )[0]


def layer_iterate_forward(
    layer,
    concat_layer,
    x,
    encoder_out,
    self_attn_mask,
    decoder_padding_mask,
    prev_output_tokens,
    nonspecial_mask,
    max_iters,
    decoder,
):
    bsz = prev_output_tokens.shape[0]

    states = {
        "x": x,
        "encoder_out": encoder_out,
        "self_attn_mask": self_attn_mask,
        "decoder_padding_mask": decoder_padding_mask,
        "prev_output_tokens": prev_output_tokens,
        "nonspecial_mask": nonspecial_mask,
        "sent_idx": utils.new_arange(prev_output_tokens, bsz),
    }
    decoder_out = {
        "x": [[] for _ in range(bsz)],
        "logits": [[] for _ in range(bsz)],
        "iters": [[] for _ in range(bsz)],
    }

    for iters in range(max_iters):
        concat_x = decoder.concat_tokens(
            concat_layer=concat_layer,
            hiddens=states["x"],
            tokens=states["prev_output_tokens"],
        )
        x = layer_single_forward(
            layer=layer,
            x=concat_x,
            kv=concat_x,
            encoder_out=states["encoder_out"],
            self_attn_mask=states["self_attn_mask"],
            decoder_padding_mask=states["decoder_padding_mask"],
        )
        logits = decoder.forward_output(x)  # T x B
        logits_B = logits.transpose(0, 1)

        tokens = logits_B.argmax(-1)  # B x T
        tokens = states["prev_output_tokens"].masked_scatter(
            states["nonspecial_mask"], tokens[states["nonspecial_mask"]]
        )
        terminated = (tokens == states["prev_output_tokens"]).all(1)
        if iters == max_iters - 1:
            terminated.fill_(1)
        not_terminated = ~terminated

        finalized_idxs = states["sent_idx"][terminated]
        finalized_x = x[:, terminated]  # T x B
        finalized_logits = logits[:, terminated]  # T x B
        for i, idx in enumerate(finalized_idxs):
            decoder_out["x"][idx] = finalized_x[:, i]
            decoder_out["logits"][idx] = finalized_logits[:, i]
            decoder_out["iters"][idx] = iters + 1

        if terminated.all():
            assert len(states["sent_idx"][not_terminated]) == 0

            x = decoder_out.pop("x")
            logits = decoder_out.pop("logits")
            iters = decoder_out.pop("iters")

            decoder_out = {
                "x": torch.stack(x, dim=1),
                "logits": torch.stack(logits, dim=1),
                "iters": prev_output_tokens.new_tensor(iters),
            }
            return decoder_out
        else:
            states = {
                "x": states["x"][:, not_terminated],
                "encoder_out": decoder.reorder_encoder_out(
                    states["encoder_out"],
                    not_terminated.nonzero(as_tuple=False).squeeze(),
                ),
                "self_attn_mask": states["self_attn_mask"][not_terminated]
                if states["self_attn_mask"] is not None
                else None,
                "decoder_padding_mask": states["decoder_padding_mask"][not_terminated],
                "nonspecial_mask": states["nonspecial_mask"][not_terminated],
                "sent_idx": states["sent_idx"][not_terminated],
                # update prev_output_tokens
                "prev_output_tokens": tokens[not_terminated],
            }


@dataclass
class NATUnnamedConfig(FairseqDataclass):
    layer_max_iter: List[int] = field(
        default_factory=lambda: [1],
        metadata={"help": "maximum iteration numbers for each layer"},
    )
    inference_layer: int = field(default=-1, metadata={"help": "layer index (start from 0) for inference"})
    layer_loss_weights: List[float] = field(
        default_factory=lambda: [],
        metadata={"help": "weights for layerwise predictions"},
    )
    glat_schedule: Optional[str] = field(default=None, metadata={"help": "glancing sampling rate schedule"})
    fixed_glat_ratio: bool = field(default=False, metadata={"help": "fixed glancing sampling ratio"})
    random_glat_ratio: bool = field(default=False, metadata={"help": "use random glat ratio"})
    fixed_glat_layer: Optional[int] = field(default=None, metadata={"help": "fixed glancing sampling layer"})
    fixed_cmlm_ratio: Optional[float] = field(default=None, metadata={"help": "use fixed cmlm ratio"})


@register_model("layer_glat")
class NATUnnamedModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        setattr(self.decoder, "reorder_encoder_out", self.encoder.reorder_encoder_out)
        setattr(self.decoder, "nonspecial_mask", self.nonspecial_mask)

        self.weights = args.layer_loss_weights
        if len(self.weights) == 0:
            self.weights = [1 / decoder.num_layers]

        if len(self.weights) == 1:
            self.weights *= decoder.num_layers

        assert len(self.weights) == decoder.num_layers

        self.glat_scheduler = LinearScheduler(getattr(args, "glat_schedule", None))

        self.iter_meters = [meters.AverageMeter(round=2) for _ in range(decoder.num_layers + 1)]

        from collections import defaultdict

        self._output_logging = defaultdict(int)

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        gen_parser_from_dataclass(parser, NATUnnamedConfig())

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATUnnamedDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @torch.no_grad()
    def glancing_sampling(self, encoder_out, prev_output_tokens, tgt_tokens, ratio):
        tgt_tokens_mask = self.nonspecial_mask(tgt_tokens)
        tgt_lens = tgt_tokens_mask.sum(-1, keepdim=True)

        if not getattr(self.args, "fixed_glat_ratio", False):
            with utils.model_eval(self.decoder):
                word_ins_out = self.decoder(
                    normalize=False,
                    prev_output_tokens=prev_output_tokens,
                    encoder_out=encoder_out,
                )[-1]["logits"]

            pred_scores, pred_tokens = word_ins_out.max(-1)

            match_pos = (pred_tokens == tgt_tokens) & tgt_tokens_mask
            match_num = match_pos.sum(-1, keepdim=True)
            # the more the matched number, the less the probs to keep target
            keep_probs = (tgt_lens - match_num) / tgt_lens * ratio
        else:
            keep_probs = ratio

        keep_mask = (torch.rand_like(prev_output_tokens.float()) < keep_probs).bool()

        glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_mask, 0) + tgt_tokens.masked_fill(~keep_mask, 0)

        glat_ratio = utils.item(keep_mask.sum() / tgt_lens.sum())

        length_factor = 1 - glat_ratio

        glat_logs = {"glat_ratio@sample": glat_ratio}
        return glat_prev_output_tokens, length_factor, glat_logs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        length_loss_factor = self.decoder.length_loss_factor

        # decoding
        ret = {"output_logs": {}}

        if getattr(self.args, "fixed_glat_layer", None) is not None:
            self.decoder.random_layer = self.args.fixed_glat_layer
        else:
            self.decoder.random_layer = torch.randint(1, self.decoder.num_layers + 1, size=(1,)).item()

        if getattr(self.args, "random_glat_ratio", False):
            glat_ratio = prev_output_tokens.float().uniform_()
            glat_flag = True
        else:
            glat_ratio = self.glat_scheduler.get_value(self.get_num_updates())
            glat_flag = glat_ratio is not None and glat_ratio > 0

        if glat_flag:
            prev_output_tokens, length_ratio, glat_logs = self.glancing_sampling(
                encoder_out,
                prev_output_tokens,
                tgt_tokens,
                ratio=glat_ratio,
            )
            length_loss_factor *= length_ratio
            ret["output_logs"].update(glat_logs)

        self.decoder.random_layer = None

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens,
        )

        word_ins_mask = prev_output_tokens.eq(self.unk)

        ret["length"] = {
            "out": length_out,
            "tgt": length_tgt,
            "factor": length_loss_factor,
        }

        for idx, output in enumerate(word_ins_out):
            ret[f"word_ins_{idx + 1}"] = {
                "out": output["logits"],
                "tgt": tgt_tokens,
                "mask": output.get("mask", word_ins_mask),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "report_accuracy": True,
                "factor": self.weights[idx],
            }

        return ret

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        output_masks = self.nonspecial_mask(output_tokens)

        # execute the decoder
        inner_states = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
            return_all_hiddens=True,
        )[1]["inner_states"]

        assert len(inner_states) == self.decoder.inference_layer + 1

        output_scores = inner_states[-1]["scores"]
        output_tokens = inner_states[-1]["tokens"]

        bsz = output_scores.size(0)

        if step == 1:
            self.output_logging.clear()

        iters_total = 0
        for i in range(len(inner_states)):
            if history is not None:
                history.append(
                    {
                        "tokens": inner_states[i]["tokens"],
                        "scores": inner_states[i]["scores"],
                        "steps": inner_states[i]["iters"],
                    }
                )
            iters = utils.item(inner_states[i]["iters"].sum())
            self.output_logging[f"iters_{i + 1}@nsentences"] += iters
            self.iter_meters[i].update(iters / bsz, bsz)

            iters_total += iters

        iters_total /= len(inner_states)
        self.output_logging["iters_total@nsentences"] += iters_total
        self.iter_meters[-1].update(iters_total / bsz, bsz)

        if step < max_step:
            if getattr(self.args, "fixed_cmlm_ratio", None) is not None:
                cmlm_ratio = self.args.fixed_cmlm_ratio
            else:
                cmlm_ratio = 1 - step / max_step
            skeptical_mask = _skeptical_unmasking(
                output_scores,
                self.nonspecial_mask(output_tokens),
                cmlm_ratio,
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def generation_string(self):
        output = ""
        for i, meter in enumerate(self.iter_meters):
            if i == len(self.iter_meters) - 1:
                output += f"iters_total: {meter.smoothed_value} | "
            else:
                output += f"iters_{i + 1}: {meter.smoothed_value} | "
        return output


class NATUnnamedDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.inference_layer = args.inference_layer
        if self.inference_layer < 0:
            self.inference_layer += self.num_layers

        self.layer_max_iter = args.layer_max_iter
        if len(self.layer_max_iter) == 1:
            self.layer_max_iter *= self.num_layers
        self.layer_max_iter[0] = 1

        self.layerwise_concat_linears = nn.ModuleList(
            [
                nn.Linear(
                    args.decoder.embed_dim * 2,
                    args.decoder.embed_dim,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.random_layer = None

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

    def forward_output(self, x):
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        return self.output_layer(x)

    def concat_tokens(self, concat_layer, hiddens, tokens):
        x = self.embed_tokens(tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        x = x.transpose(0, 1)
        concat_x = concat_layer(torch.cat([hiddens, x], dim=-1))
        return concat_x

    @ensemble_decoder
    def forward(
        self,
        normalize,
        encoder_out,
        prev_output_tokens,
        step=1,
        return_all_hiddens=False,
        **kwargs,
    ):
        embedding_copy = False
        if self.src_embedding_copy and prev_output_tokens.eq(self.unk).any():
            embedding_copy = True

        feature_list, inner_states = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=embedding_copy,
            step=step,
            **kwargs,
        )

        if not return_all_hiddens:
            return feature_list
        return feature_list, inner_states

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        tgt_tokens=None,
        glat_ratio=None,
        step=1,
        **unused,
    ):
        tokens = prev_output_tokens.masked_fill(self.nonspecial_mask(prev_output_tokens), self.unk)

        # embedding
        if embedding_copy:
            copied_embedding = self.forward_copying_source(encoder_out, tokens)
            x, decoder_padding_mask = self.forward_embedding(tokens, states=copied_embedding)
        else:
            x, decoder_padding_mask = self.forward_embedding(tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        tokens = prev_output_tokens

        attns = []
        inner_states = []
        feature_list = []

        nonspecial_mask = prev_output_tokens.eq(self.unk)
        real_pos_mask = self.nonspecial_mask(prev_output_tokens)

        # if tgt_tokens is not None:
        #     nonspecial_mask = prev_output_tokens.eq(self.unk)
        # else:
        #     nonspecial_mask = self.nonspecial_mask(prev_output_tokens)

        def _scores_and_tokens(_logits, log_softmax=False):  # B x T
            assert _logits.shape[:2] == prev_output_tokens.shape
            if log_softmax:
                _scores, _tokens = _logits.log_softmax(-1).max(-1)
            else:
                _scores, _tokens = _logits.max(-1)
            _scores = _scores.masked_fill(~real_pos_mask, 0)
            _tokens = prev_output_tokens.masked_scatter(nonspecial_mask, _tokens[nonspecial_mask])
            return _scores, _tokens

        random_layer = self.random_layer if self.random_layer is not None else self.num_layers

        if tgt_tokens is not None:
            for i, layer in enumerate(self.layers[:random_layer]):
                concat_x = self.concat_tokens(
                    concat_layer=self.layerwise_concat_linears[i],
                    hiddens=x,
                    tokens=tokens,
                )

                x = layer_single_forward(
                    layer=layer,
                    x=concat_x,
                    kv=concat_x,
                    encoder_out=encoder_out,
                    self_attn_mask=None,
                    decoder_padding_mask=decoder_padding_mask,
                )
                logits = self.forward_output(x)  # T x B
                logits_B = logits.transpose(0, 1)

                feature_list.append({"logits": logits_B})  # B x T

                scores, tokens = _scores_and_tokens(logits_B)

        else:
            for i, layer in enumerate(self.layers[:random_layer]):
                if self.layer_max_iter[i] == 1:
                    concat_x = self.concat_tokens(
                        concat_layer=self.layerwise_concat_linears[i],
                        hiddens=x,
                        tokens=tokens,
                    )
                    x = layer_single_forward(
                        layer=layer,
                        x=concat_x,
                        kv=concat_x,
                        encoder_out=encoder_out,
                        self_attn_mask=None,
                        decoder_padding_mask=decoder_padding_mask,
                    )
                    logits = self.forward_output(x)
                    decoder_out = {"iters": prev_output_tokens.new_ones(prev_output_tokens.size(0))}
                else:
                    decoder_out = layer_iterate_forward(
                        layer=layer,
                        concat_layer=self.layerwise_concat_linears[i],
                        x=x,
                        encoder_out=encoder_out,
                        self_attn_mask=None,
                        decoder_padding_mask=decoder_padding_mask,
                        prev_output_tokens=tokens,
                        nonspecial_mask=nonspecial_mask,
                        max_iters=self.layer_max_iter[i],
                        decoder=self,
                    )
                    x = decoder_out["x"]  # T x B
                    logits = decoder_out["logits"]  # T x B

                logits_B = logits.transpose(0, 1)
                feature_list.append({"logits": logits_B})

                scores, tokens = _scores_and_tokens(logits_B, log_softmax=i == self.inference_layer)

                decoder_out.update({"scores": scores, "tokens": tokens})
                inner_states.append(decoder_out)

                if tgt_tokens is None and i == self.inference_layer:
                    break

        return feature_list, {"attns": attns, "inner_states": inner_states}


@register_model_architecture("layer_glat", "layer_glat")
def glancing_transformer_base_architecture(args):
    base_architecture(args)


@register_model_architecture("layer_glat", "layer_glat_wmt_en_de")
def glancing_transformer_wmt_en_de(args):
    glancing_transformer_base_architecture(args)


@register_model_architecture("layer_glat", "layer_glat_iwslt_de_en")
def glancing_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    glancing_transformer_base_architecture(args)
