# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import numpy as np
import torch
from fairseq import meters, utils

DecoderOut = namedtuple(
    "IterativeRefinementDecoderOut",
    ["output_tokens", "output_scores", "attn", "step", "max_step", "history"],
)


def _merge_beam_results(beam_results):
    """
    See `"Candidate Soups: Fusing Candidate Results Improves Translation Quality for Non-Autoregressive Translation"
    (Zheng et al., 2023) <https://arxiv.org/abs/2301.11503>`_.
    """
    remain_tokens = [beam[0]["tokens"].tolist() for beam in beam_results]
    remain_scores = [beam[0]["positional_scores"].tolist() for beam in beam_results]
    beam_size = len(remain_tokens)

    segment_tokens = [[] for _ in range(beam_size)]
    segment_scores = [[] for _ in range(beam_size)]

    for token in list(remain_tokens[0]):
        if all([token in beam for beam in remain_tokens]):
            for i in range(beam_size):
                index = remain_tokens[i].index(token) + 1

                segment_tokens[i].append(remain_tokens[i][:index])
                remain_tokens[i] = remain_tokens[i][index:]

                segment_scores[i].append(remain_scores[i][:index])
                remain_scores[i] = remain_scores[i][index:]

    segment_tokens = list(zip(*segment_tokens))
    segment_scores = list(zip(*segment_scores))

    merged_tokens, merged_scores = [], []
    for tokens, scores in zip(segment_tokens, segment_scores):
        index = np.argmax([np.mean(s) for s in scores])
        merged_tokens.extend(tokens[index])
        merged_scores.extend(scores[index])

    return {
        "steps": max(beam[0]["steps"] for beam in beam_results),
        "tokens": beam_results[0][0]["tokens"].new_tensor(merged_tokens),
        "positional_scores": beam_results[0][0]["positional_scores"].new_tensor(merged_scores),
        "score": beam_results[0][0]["score"].new_tensor(np.mean(merged_scores)),
        "hypo_attn": None,
        "alignment": None,
    }


class IterativeRefinementGenerator(object):
    def __init__(
        self,
        tgt_dict,
        models=None,
        eos_penalty=0.0,
        max_iter=10,
        max_ratio=2,
        beam_size=1,
        decoding_format=None,
        retain_dropout=False,
        adaptive=True,
        retain_history=False,
        reranking=False,
        len_penalty=1.0,
        length_beam_topk=False,
        decode_with_oracle_length=False,
        remove_consecutive_repetition=False,
        merge_beam_results=False,
    ):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
            eos_penalty: if > 0.0, it penalized early-stopping in decoding
            max_iter: maximum number of refinement iterations
            max_ratio: generate sequences of maximum length ax, where x is the source length
            decoding_format: decoding mode in {'unigram', 'ensemble', 'vote', 'dp', 'bs'}
            retain_dropout: retaining dropout in the inference
            adaptive: decoding with early stop
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.eos_penalty = eos_penalty
        self.max_iter = max_iter
        self.max_ratio = max_ratio
        self.beam_size = beam_size
        self.reranking = reranking
        self.decoding_format = decoding_format
        self.retain_dropout = retain_dropout
        self.retain_history = retain_history
        self.adaptive = adaptive
        self.models = models

        self.len_penalty = len_penalty
        self.length_beam_topk = length_beam_topk
        self.decode_with_oracle_length = decode_with_oracle_length
        self.remove_consecutive_repetition = remove_consecutive_repetition
        self.merge_beam_results = merge_beam_results

        self.iter_meter = meters.AverageMeter(round=2)

        self._output_logging = {}
        self._generation_string_suffix = ""

    def generation_string(self):
        string = (
            f"lenpen: {self.len_penalty} | "
            + f"remove_rep: {self.remove_consecutive_repetition} | "
            + f"merge_beam: {self.merge_beam_results} | "
            + f"average_iters: {self.iter_meter.smoothed_value} | "
        )
        return string + self._generation_string_suffix

    def generate_batched_itr(
        self,
        data_itr,
        maxlen_a=None,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.models,
                    sample,
                    prefix_tokens=sample["target"][:, :prefix_size] if prefix_size > 0 else None,
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError("Constrained decoding with the IterativeRefinementGenerator is not supported")

        # TODO: iterative refinement generator does not support ensemble for now.
        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None
        if self.reranking:
            assert len(models) > 1, "Assuming the last checkpoint is the reranker"
            assert self.beam_size > 1, "Reranking requires multiple translation for each example"

            reranker = models[-1]
            models = models[:-1]

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(model.__class__.__name__)
            model.enable_ensemble(models)

        # TODO: better encoder inputs?
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths])

        initialize = {"beam_size": self.beam_size, "topk": self.length_beam_topk}
        if self.decode_with_oracle_length:
            assert self.beam_size == 1, "decode with target length only support length beam > 1"
            initialize["length_tgt"] = sample["target"].ne(self.pad).sum(-1)

        prev_decoder_out = model.initialize_output_tokens(encoder_out, src_tokens, **initialize)

        if self.beam_size > 1:
            assert model.allow_length_beam, "{} does not support decoding with length beam.".format(
                model.__class__.__name__
            )

            # regenerate data based on length-beam
            length_beam_order = utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            encoder_out = model.encoder.reorder_encoder_out(encoder_out, length_beam_order)
            assert prev_decoder_out.output_tokens.size(0) == bsz * self.beam_size

            bsz = bsz * self.beam_size

        sent_idxs = torch.arange(bsz)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.retain_history:
            prev_decoder_out = prev_decoder_out._replace(history=[{"tokens": prev_output_tokens}])

        finalized = [[] for _ in range(bsz)]

        def is_a_loop(x, y, s, a):
            b, l_x, l_y = x.size(0), x.size(1), y.size(1)
            if l_x > l_y:
                y = torch.cat([y, x.new_zeros(b, l_x - l_y).fill_(self.pad)], 1)
                s = torch.cat([s, s.new_zeros(b, l_x - l_y)], 1)
                if a is not None:
                    a = torch.cat([a, a.new_zeros(b, l_x - l_y, a.size(2))], 1)
            elif l_x < l_y:
                x = torch.cat([x, y.new_zeros(b, l_y - l_x).fill_(self.pad)], 1)
            return (x == y).all(1), y, s, a

        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn, post_edit=False):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]

            if post_edit and self.remove_consecutive_repetition:
                tokens, counts = torch.unique_consecutive(tokens, return_counts=True)
                if scores is not None and tokens.shape[0] != scores.shape[0]:
                    new_scores = scores.new_zeros(*tokens.shape)
                    cumsum = [0] + counts.cumsum(-1).tolist()
                    for i, (start, stop) in enumerate(zip(cumsum[:-1], cumsum[1:])):
                        new_scores[i] = torch.max(scores[start:stop])
                    scores = new_scores.clone()

            if scores is not None:
                score = scores.sum() / scores.shape[0] ** self.len_penalty

            if prev_out_attn is None or self.remove_consecutive_repetition:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }

        for step in range(1, self.max_iter + 1):
            decoder_options = {
                "eos_penalty": self.eos_penalty,
                "max_ratio": self.max_ratio,
                "decoding_format": self.decoding_format,
                "src_tokens": src_tokens,
            }
            prev_decoder_out = prev_decoder_out._replace(step=step, max_step=self.max_iter)

            decoder_out = model.forward_decoder(prev_decoder_out, encoder_out, **decoder_options)

            if self.adaptive:
                # terminate if there is a loop
                terminated, out_tokens, out_scores, out_attn = is_a_loop(
                    prev_output_tokens,
                    decoder_out.output_tokens,
                    decoder_out.output_scores,
                    decoder_out.attn,
                )
                decoder_out = decoder_out._replace(
                    output_tokens=out_tokens,
                    output_scores=out_scores,
                    attn=out_attn,
                )

            else:
                terminated = decoder_out.output_tokens.new_zeros(decoder_out.output_tokens.size(0)).bool()

            if step == self.max_iter:  # reach last iteration, terminate
                terminated.fill_(1)

            # collect finalized sentences
            finalized_idxs = sent_idxs[terminated.to(sent_idxs.device)]
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated]
            finalized_attn = (
                None if (decoder_out.attn is None or decoder_out.attn.size(0) == 0) else decoder_out.attn[terminated]
            )

            if self.retain_history:
                finalized_history = [{k: v[terminated] for k, v in h.items()} for h in decoder_out.history]

            for i in range(finalized_idxs.size(0)):
                finalized[finalized_idxs[i]] = [
                    finalized_hypos(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i],
                        None if finalized_attn is None else finalized_attn[i],
                        post_edit=True,
                    )
                ]

                if self.retain_history:
                    finalized[finalized_idxs[i]][0]["history"] = []
                    for j in range(len(finalized_history)):
                        finalized[finalized_idxs[i]][0]["history"].append(
                            finalized_hypos(
                                j,
                                finalized_history[j]["tokens"][i],
                                finalized_history[j]["scores"][i] if "scores" in finalized_history[j] else None,
                                None,
                                post_edit=False,
                            )
                        )

            # check if all terminated
            if terminated.sum() == terminated.size(0):
                break

            # for next step
            not_terminated = ~terminated
            prev_decoder_out = model.filter_terminated(decoder_out, not_terminated)
            encoder_out = model.encoder.reorder_encoder_out(
                encoder_out, not_terminated.nonzero(as_tuple=False).squeeze()
            )
            sent_idxs = sent_idxs[not_terminated.to(sent_idxs.device)]
            prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.beam_size > 1:
            if reranker is not None:
                finalized = self.rerank(
                    reranker,
                    finalized,
                    [
                        src_tokens.masked_fill(src_tokens.eq(self.bos), self.pad)[:, 1:],
                        src_lengths - 1,
                    ],
                    self.beam_size,
                )

            # aggregate information from length beam
            if not self.merge_beam_results:
                finalized = [
                    finalized[
                        np.argmax([finalized[self.beam_size * i + j][0]["score"].cpu() for j in range(self.beam_size)])
                        + self.beam_size * i
                    ]
                    for i in range(len(finalized) // self.beam_size)
                ]
            else:
                finalized = [
                    [_merge_beam_results([finalized[self.beam_size * i + j] for j in range(self.beam_size)])]
                    for i in range(len(finalized) // self.beam_size)
                ]

        self.output_logging.clear()

        total_iters = utils.item(sum(x[0]["steps"] for x in finalized))
        # used for validation during training
        self.output_logging["iters@nsentences"] = total_iters
        # used for inference
        self.iter_meter.update(total_iters / len(finalized), len(finalized))

        self.output_logging.update(model.output_logging)

        if hasattr(model, "generation_string"):
            self._generation_string_suffix = model.generation_string()

        return finalized

    def rerank(self, reranker, finalized, encoder_input, beam_size):
        def rebuild_batch(finalized):
            finalized_tokens = [f[0]["tokens"] for f in finalized]
            finalized_maxlen = max(f.size(0) for f in finalized_tokens)
            final_output_tokens = (
                finalized_tokens[0].new_zeros(len(finalized_tokens), finalized_maxlen).fill_(self.pad)
            )
            for i, f in enumerate(finalized_tokens):
                final_output_tokens[i, : f.size(0)] = f
            return final_output_tokens

        final_output_tokens = rebuild_batch(finalized)
        final_output_tokens[:, 0] = self.eos  # autoregressive model assumes starting with EOS

        reranker_encoder_out = reranker.encoder(*encoder_input)
        length_beam_order = (
            utils.new_arange(
                final_output_tokens,
                beam_size,
                reranker_encoder_out["encoder_out"][0].size(1),
            )
            .t()
            .reshape(-1)
        )
        reranker_encoder_out = reranker.encoder.reorder_encoder_out(reranker_encoder_out, length_beam_order)
        reranking_scores = reranker.get_normalized_probs(
            reranker.decoder(final_output_tokens[:, :-1], reranker_encoder_out),
            True,
            None,
        )
        reranking_scores = reranking_scores.gather(2, final_output_tokens[:, 1:, None])
        reranking_masks = final_output_tokens[:, 1:].ne(self.pad)
        reranking_scores = reranking_scores[:, :, 0].masked_fill_(~reranking_masks, 0)

        for i in range(len(finalized)):
            finalized[i][0]["positional_scores"][1:] = reranking_scores[i][reranking_masks[i]]
            finalized[i][0]["score"] = finalized[i][0]["positional_scores"].mean()

        return finalized

    @property
    def output_logging(self):
        return self._output_logging
