import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq_plugin.utils import LinearScheduler

from .ctc_utils import CTCLoss, longest_common_subsequence
from .nonautoregressive_transformer import (
    NATransformerDecoder,
    NATransformerModel,
    base_architecture,
    ensemble_decoder,
)

logger = logging.getLogger(__name__)

CTC_BLANK = "<blank>"
MASK_STRATEGY = ChoiceEnum(["random"])


@dataclass
class GlancingTransformerModelConfig(FairseqDataclass):
    # Glancing Transformer
    glat_schedule: Optional[str] = field(default=None, metadata={"help": "glancing sampling rate schedule"})
    glat_mask_strategy: MASK_STRATEGY = field(
        default="random", metadata={"help": "mask strategy for glancing sampling"}
    )
    adaptive_length: bool = field(
        default=False, metadata={"help": "use adaptive length factor based on the number of masked tokens"}
    )
    # CTC-based configs
    ctc: bool = field(default=False, metadata={"help": "use ctc loss, if glat is enabled, then use glat + ctc"})
    ctc_upsample_scale: float = field(default=2.0, metadata={"help": "upsampling scale for the CTC input"})
    imputered_ctc: bool = field(default=False, metadata={"help": "use imputer loss to train CTC model"})
    # Layer-wise Prediction
    layerwise_preds: bool = field(default=False, metadata={"help": "enable layer-wise prediction and training"})
    inference_layer: int = field(default=-1, metadata={"help": "layer index used for inference, starts from 0"})
    train_hard_embed: bool = field(
        default=False, metadata={"help": "layer-wise training with hard argmax embedding. Implies --train-hard-embed"}
    )
    inference_hard_embed: bool = field(
        default=False, metadata={"help": "layer-wise inference with hard argmax embedding."}
    )


@register_model("glancing_transformer")
class GlancingTransformerModel(NATransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.glat_scheduler = LinearScheduler(args.glat_schedule)
        self.adaptive_length = getattr(args, "adaptive_length", False)

        self.use_ctc = getattr(args, "ctc", False)
        if self.use_ctc:
            try:
                self.blank_idx = decoder.dictionary.indices[CTC_BLANK]
            except KeyError:
                raise KeyError("No CTC blank token is found, please add it with --extra-special-symbols")

            self.ctc_loss = CTCLoss(
                blank_idx=self.blank_idx,
                label_smoothing=args.label_smoothing,
                imputered_ctc=getattr(args, "imputered_ctc", False),
            )
            self.longest_common_subsequence = longest_common_subsequence

    @property
    def allow_length_beam(self):
        return False if self.use_ctc else True

    @property
    def allow_ensemble(self):
        return False if self.use_ctc else True

    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)
        gen_parser_from_dataclass(parser, GlancingTransformerModelConfig())

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = GLATNATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def max_positions(self):
        encoder = self.encoder.max_positions()
        decoder = self.decoder.max_positions()
        if self.use_ctc:
            decoder = min(decoder, int(encoder * self.args.ctc_upsample_scale))
        return encoder, decoder

    @torch.no_grad()
    def glancing_sampling(self, encoder_out, prev_output_tokens, tgt_tokens):
        with utils.model_eval(self):
            word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )
        pred_scores, pred_tokens = word_ins_out[-1].max(-1)

        tgt_tokens_mask = self.nonspecial_mask(tgt_tokens)
        tgt_lens = tgt_tokens_mask.sum(-1, keepdim=True)

        if self.use_ctc:
            lcs = self.longest_common_subsequence(tgt_tokens, pred_tokens, self.pad)
            match_num = self.nonspecial_mask(lcs).sum(-1, keepdim=True)
        else:
            match_pos = (pred_tokens == tgt_tokens) & tgt_tokens_mask
            match_num = match_pos.sum(-1, keepdim=True)

        if self.args.glat_mask_strategy == "random":
            ratio = self.glat_scheduler.get_value(self.get_num_updates())
            # the more the matched number, the less the probs to keep target
            keep_probs = (tgt_lens - match_num) / tgt_lens * ratio
            keep_mask = (torch.rand_like(prev_output_tokens.float()) < keep_probs).bool()
        else:
            raise ValueError(f"Unrecognized glancing mask strategy: {self.args.glat_mask_strategy}")

        if self.use_ctc:
            # CTC use imputered tgt_tokens
            tgt_tokens = self.ctc_loss.best_alignment(
                word_ins_out[-1],
                tgt_tokens,
                input_mask=prev_output_tokens.ne(self.pad),
                target_mask=tgt_tokens.ne(self.pad),
            )
            # only replace these real tokens
            keep_mask = keep_mask & tgt_tokens.ne(self.blank_idx) & self.nonspecial_mask(tgt_tokens)

        glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_mask, 0) + tgt_tokens.masked_fill(~keep_mask, 0)

        glat_ratio = utils.item(keep_mask.sum() / tgt_lens.sum())

        length_factor = 1
        if not self.use_ctc and self.adaptive_length:
            # if glat_ratio = 0, no glancing, so we use scale=1.
            length_factor = 1 - glat_ratio

        glat_logs = {
            "_glat@total": utils.item(tgt_lens.sum()),
            "_glat@n_correct": utils.item(match_num.sum()),
            "glat_ratio@nsamples": glat_ratio,
        }
        return glat_prev_output_tokens, length_factor, glat_logs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        if self.use_ctc:
            length_out = None
            length_tgt = (self.args.ctc_upsample_scale * src_lengths).long()
            prev_output_tokens = self._get_initial_output_tokens(src_tokens, length_tgt)
        else:
            length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
            length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
        length_loss_factor = self.decoder.length_loss_factor

        # decoding
        ret = {"output_logs": {}}
        if self.glat_scheduler.get_value(self.get_num_updates()):
            prev_output_tokens, length_ratio, glat_logs = self.glancing_sampling(
                encoder_out, prev_output_tokens, tgt_tokens
            )
            length_loss_factor *= length_ratio
            ret["output_logs"].update(glat_logs)

        if length_out is not None:
            ret["length"] = {
                "out": length_out,
                "tgt": length_tgt,
                "factor": length_loss_factor,
            }

        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        if self.use_ctc:
            loss, ctc_loss = self.ctc_loss(
                word_ins_out[-1],
                tgt_tokens,
                input_mask=prev_output_tokens.ne(self.pad),
                target_mask=tgt_tokens.ne(self.pad),
            )
            ret["ctc"] = {"loss": loss, "nll_loss": ctc_loss}
        else:
            if self.decoder.layerwise_preds:
                for idx, output in enumerate(word_ins_out, start=1):
                    ret[f"word_ins_{idx}"] = {
                        "out": output,
                        "tgt": tgt_tokens,
                        "mask": prev_output_tokens.eq(self.unk),
                        "ls": self.args.label_smoothing,
                        "nll_loss": True,
                        "report_accuracy": idx == len(word_ins_out),
                        "factor": 1 / len(word_ins_out),
                    }
            else:
                ret["word_ins"] = {
                    "out": word_ins_out[-1],
                    "tgt": tgt_tokens,
                    "mask": prev_output_tokens.eq(self.unk),
                    "ls": self.args.label_smoothing,
                    "nll_loss": True,
                    "report_accuracy": True,
                }

        return ret

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        if self.use_ctc:
            output_masks = output_tokens.ne(self.pad)
        else:
            output_masks = output_tokens.eq(self.unk)

        word_ins_out = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )
        _scores, _tokens = word_ins_out[self.args.inference_layer].max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append({"tokens": output_tokens.clone(), "scores": output_scores.clone()})

        if self.use_ctc:
            assert step == 1
            ctc_inverse_mask = self.ctc_loss.ctc_inverse_mask(output_tokens, batch_first=True)
            output_tokens = output_tokens.masked_fill(ctc_inverse_mask, self.pad)
            output_scores = output_scores.masked_fill(ctc_inverse_mask, 0)

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens, beam_size=1, topk=False, length_tgt=None):
        if self.use_ctc:
            length_tgt = (self.args.ctc_upsample_scale * src_tokens.ne(self.pad).sum(-1)).long()
        return super().initialize_output_tokens(encoder_out, src_tokens, beam_size, topk, length_tgt)


class GLATNATransformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.layerwise_preds = getattr(args, "layerwise_preds", False)
        if self.layerwise_preds:
            self.layerwise_concat_linears = nn.ModuleList(
                [
                    nn.Linear(
                        args.decoder.embed_dim * 2,
                        args.decoder.embed_dim,
                    )
                    for _ in range(args.decoder.layers - 1)
                ]
            )

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
            **kwargs,
        )

        if normalize:
            feature_list = [torch.log_softmax(feature, -1) for feature in feature_list]

        if not return_all_hiddens:
            return feature_list
        return feature_list, inner_states

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        **unused,
    ):
        # embedding
        if embedding_copy:
            copied_embedding = self.forward_copying_source(
                encoder_out,
                prev_output_tokens,
            )
            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                states=copied_embedding,
            )
        else:
            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attns = []
        inner_states = [x]
        feature_list = []

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
            )
            inner_states.append(x)
            attns.append(attn)

            if self.layerwise_preds:
                layer_logits = self.output_layer(x if self.project_out_dim is None else self.project_out_dim(x))
                feature_list.append(layer_logits.transpose(0, 1))

                if getattr(self.args, "train_hard_embed", False) or (
                    not self.training and getattr(self.args, "inference_hard_embed", False)
                ):
                    # if hard embed is used during training, we will use it
                    # at both the training and inference stages
                    layer_x = self.embed_tokens(layer_logits.argmax(-1))
                else:
                    layer_x = torch.einsum(
                        "aij,jk->aik",
                        torch.softmax(layer_logits, dim=-1),
                        self.embed_tokens.weight,
                    )

                if i < self.num_layers - 1:
                    x = self.layerwise_concat_linears[i](torch.cat([x, layer_x], dim=-1))

        if len(feature_list) == 0:
            output = self.output_layer(x if self.project_out_dim is None else self.project_out_dim(x))
            feature_list.append(output.transpose(0, 1))

        return feature_list, {"attns": attns, "inner_states": inner_states}


@register_model_architecture("glancing_transformer", "glancing_transformer")
def glancing_transformer_base_architecture(args):
    base_architecture(args)


@register_model_architecture("glancing_transformer", "glancing_transformer_wmt_en_de")
def glancing_transformer_wmt_en_de(args):
    glancing_transformer_base_architecture(args)


@register_model_architecture("glancing_transformer", "glancing_transformer_iwslt_de_en")
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
