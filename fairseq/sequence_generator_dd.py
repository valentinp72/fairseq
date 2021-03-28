# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock
from .sequence_generator import SequenceGenerator, EnsembleModel


class SequenceGeneratorDD(SequenceGenerator):
    def __init__(self, models, tgt_dict, **kwargs):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__(models, tgt_dict, **kwargs)
        if isinstance(models, EnsembleModelDD):
            self.model = models
        else:
            self.model = EnsembleModelDD(models)
        self.search = (search.DualBeamSearch(tgt_dict))

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)
        # logging.info(f'encoder_outs: {encoder_outs[0]["encoder_out"][0].shape}')  # T x bsz x encoder_dim

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None
        # logging.info(f'encoder_outs after reorder: {encoder_outs[0]["encoder_out"][0].shape}') # T x (bsz x beam_size) x encoder_dim

        # initialize buffers
        scores = [torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float() 
            for i in range(2)] # +1 for eos; pad is never chosen for scoring     
        tokens = [torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad) for i in range(2)]  # +2 for eos and pad
        for i in range(2):
            tokens[i][:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[List[Tensor]] = [None, None]

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = tuple(
            [torch.zeros(bsz, beam_size).to(src_tokens).eq(-1) for i in range(2)]
        )  # forward and backward-compatible False mask
        # logging.info(f'cands_to_ignore: {cands_to_ignore}')

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step
        # logging.info(f'finalized: {finalized}')

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining
        # logging.info(f'finished: {finished}')

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = tuple(
            [(torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens[i])
            .to(src_tokens.device) for i in range(2)]
        )
        cand_offsets = tuple(
            [torch.arange(0, cand_size).type_as(tokens[i]).to(src_tokens.device) 
            for i in range(2)]
            )
        # logging.info(f'bbsz_offsets: {bbsz_offsets}')
        # logging.info(f'cand_offsets: {cand_offsets}')

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens[0])
        # logging.info(f'original_batch_idxs: {original_batch_idxs}')

        for step in range(max_len + 1):  # one extra step for EOS marker
            # logging.info(f'****** step = {step}')
            # logging.info(f'****** reorder_state = {reorder_state}')
            # logging.info(f'****** incremental_states = {incremental_states}')
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
                # logging.info(f'encoder_outs after reorder: {encoder_outs[0]["encoder_out"][0].shape}') # T x (bsz x beam_size) x encoder_dim
            
            # logging.info(f'incremental_states: {incremental_states}')
            lprobs, avg_attn_scores = self.model.forward_decoder(
                [tokens[i][:, : step + 1] for i in range(2)],
                encoder_outs,
                incremental_states,
                self.temperature,
            )
            # logging.info(f'lprobs: {lprobs}')
            # logging.info(f'avg_attn_scores: {avg_attn_scores}')

            lprobs = list(lprobs)
            if self.lm_model is not None:
                lm_out = []
                probs = []
                for i in range(2):
                    lm_out.append(self.lm_model(tokens[i][:, : step + 1]))
                    probs.append(self.lm_model.get_normalized_probs(
                        lm_out[i], log_probs=True, sample=None
                    ))
                    probs[i] = probs[i][:, -1, :] * self.lm_weight
                    lprobs[i] += probs[i]

            for i in range(2):
                lprobs[i][lprobs[i] != lprobs[i]] = torch.tensor(-math.inf).to(lprobs[i])

                lprobs[i][:, self.pad] = -math.inf  # never select pad
                lprobs[i][:, self.unk] -= self.unk_penalty  # apply unk penalty

                # handle max length constraint
                if step >= max_len:
                    lprobs[i][:, : self.eos] = -math.inf
                    lprobs[i][:, self.eos + 1 :] = -math.inf

                # handle prefix tokens (possibly with different lengths)
                if (
                    prefix_tokens[i] is not None
                    and step < prefix_tokens[i].size(1)
                    and step < max_len
                ):
                    lprobs[i], tokens[i], scores[i] = self._prefix_tokens(
                        step, lprobs[i], scores[i], tokens[i], prefix_tokens[i], beam_size
                    )
                elif step < self.min_len:
                    # minimum length constraint (does not apply if using prefix_tokens)
                    lprobs[i][:, self.eos] = -math.inf

                # Record attention scores, only support avg_attn_scores is a Tensor
                if avg_attn_scores is not None: # TODO: to check
                    if attn == [None, None]:
                        attn[i] = torch.empty(
                            bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                        ).to(scores[i])
                    attn[i][:, :, step + 1].copy_(avg_attn_scores)

                scores[i] = scores[i].type_as(lprobs[i])
            eos_bbsz_idx = [torch.empty(0).to(
                tokens[i]
            ) for i in range(2)]  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = [torch.empty(0).to(
                scores[i]
            ) for i in range(2)]  # scores of hypothesis ending with eos (finished sentences)
            # logging.info(f'eos_bbsz_idx: {eos_bbsz_idx}')
            # logging.info(f'eos_scores: {eos_scores}')

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                [lprobs[i].view(bsz, -1, self.vocab_size) for i in range(2)],
                [scores[i].view(bsz, beam_size, -1)[:, :, :step] for i in range(2)],
                [tokens[i][:, : step + 1] for i in range(2)],
                original_batch_idxs,
            )
            # logging.info(f'cand_scores.shape={[e.shape for e in cand_scores]} \n {cand_scores}')
            # logging.info(f'cand_indices.shape={[e.shape for e in cand_indices]} \n {cand_indices}')
            # logging.info(f'cand_beams.shape={[e.shape for e in cand_beams]} \n {cand_beams}')

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = [cand_beams[i].add(bbsz_offsets[i]) for i in range(2)]
            # logging.info(f'cand_bbsz_idx.shape={[e.shape for e in cand_bbsz_idx]} \n {cand_bbsz_idx}')

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = [cand_indices[i].eq(self.eos) & cand_scores[i].ne(-math.inf) for i in range(2)]
            # logging.info(f'eos_mask.shape={[e.shape for e in eos_mask]} \n {eos_mask}')
            for i in range(2):
                eos_mask[i][:, :beam_size][cands_to_ignore[i]] = torch.tensor(0).to(eos_mask[i])
            # logging.info(f'eos_mask.shape={[e.shape for e in eos_mask]} \n {eos_mask}')

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = [torch.masked_select(
                cand_bbsz_idx[i][:, :beam_size], mask=eos_mask[i][:, :beam_size]
            ) for i in range(2)]
            # logging.info(f'eos_bbsz_idx.shape={[e.shape for e in eos_bbsz_idx]} \n {eos_bbsz_idx}')

            finalized_sents: List[int] = []
            for i in range(2):
                if eos_bbsz_idx[i].numel() > 0:
                    eos_scores[i] = torch.masked_select(
                    cand_scores[i][:, :beam_size], mask=eos_mask[i][:, :beam_size]
                )
            if all([eos_bbsz_idx[i].numel() > 0 for i in range(2)]):
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)
                # logging.info(f'finalized_sents: {finalized_sents}')
                # logging.info(f'num_remaining_sent: {num_remaining_sent}')

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices[0].device
                )
                batch_mask[finalized_sents] = False
                    # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices[0].device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = [eos_mask[i][batch_idxs] for i in range(2)]
                cand_beams = [cand_beams[i][batch_idxs] for i in range(2)]
                for i in range(2):
                    bbsz_offsets[i].resize_(new_bsz, 1)
                cand_bbsz_idx = [cand_beams[i].add(bbsz_offsets[i]) for i in range(2)]
                cand_scores = [cand_scores[i][batch_idxs] for i in range(2)]
                cand_indices = [cand_indices[i][batch_idxs] for i in range(2)]

                if prefix_tokens[0] is not None and prefix_tokens[1] is not None:
                    prefix_tokens = [prefix_tokens[i][batch_idxs] for i in range(2)]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = [cands_to_ignore[i][batch_idxs] for i in range(2)]

                scores = [scores[i].view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1) for i in range(2)]
                tokens = [tokens[i].view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1) for i in range(2)]
                if None not in attn:
                    attn = [attn[i].view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn[i].size(1), -1
                    ) for i in range(2)]
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.
            for i in range(2):
                eos_mask[i][:, :beam_size] = ~((~cands_to_ignore[i]) & (~eos_mask[i][:, :beam_size]))
            active_mask = [torch.add(
                eos_mask[i].type_as(cand_offsets[i]) * cand_size,
                cand_offsets[i][: eos_mask[i].size(1)],
            ) for i in range(2)]

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos, cands_to_ignore = [None]*2, [None]*2, [None]*2
            for i in range(2):
                new_cands_to_ignore[i], active_hypos[i] = torch.topk(
                    active_mask[i], k=beam_size, dim=1, largest=False
                )

                # update cands_to_ignore to ignore any finalized hypos.
                cands_to_ignore[i] = new_cands_to_ignore[i].ge(cand_size)[:, :beam_size]
                # Make sure there is at least one active item for each sentence in the batch.
                assert (~cands_to_ignore[i]).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = [torch.gather(cand_bbsz_idx[i], dim=1, index=active_hypos[i]) for i in range(2)]
            active_scores = [torch.gather(cand_scores[i], dim=1, index=active_hypos[i]) for i in range(2)]

            active_bbsz_idx = [active_bbsz_idx[i].view(-1) for i in range(2)]
            active_scores = [active_scores[i].view(-1) for i in range(2)]

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            for i in range(2):
                tokens[i][:, : step + 1] = torch.index_select(
                    tokens[i][:, : step + 1], dim=0, index=active_bbsz_idx[i]
                )
                # Select the next token for each of them
                tokens[i].view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                    cand_indices[i], dim=1, index=active_hypos[i]
                )
                if step > 0:
                    scores[i][:, :step] = torch.index_select(
                        scores[i][:, :step], dim=0, index=active_bbsz_idx[i]
                    )
                scores[i].view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                    cand_scores[i], dim=1, index=active_hypos[i]
                )

                # Update constraints based on which candidates were selected for the next beam
                self.search.update_constraints(active_hypos[i])

                # copy attention for active hypotheses
                if None not in attn:
                    attn[i][:, :, : step + 2] = torch.index_select(
                        attn[i][:, :, : step + 2], dim=0, index=active_bbsz_idx[i]
                    )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx[0]

        # sort by score descending
        # logging.info(f'finalized \n {finalized}')
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"][0].item())+float(elem["score"][1].item()) 
                for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        # logging.info(f'step={step}, bbsz_idx={bbsz_idx}, eos_scores={eos_scores}, tokens={tokens}, scores={scores}')
        # logging.info(f'finalized={finalized}, finished={finished}, attn={attn}, src_lengths={src_lengths}')
        for i in range(2):
            assert bbsz_idx[i].numel() == eos_scores[i].numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = [tokens[i].index_select(0, bbsz_idx[i])[
            :, 1 : step + 2
        ] for i in range(2)]  # skip the first index, which is EOS

        for i in range(2):
            tokens_clone[i][:, step] = self.eos
        attn_clone = [(
            attn[i].index_select(0, bbsz_idx[i])[:, :, 1 : step + 2]
            if attn[i] is not None
            else None
        ) for i in range(2)]

        # compute scores per token position
        pos_scores = [scores[i].index_select(0, bbsz_idx[i])[:, : step + 1] for i in range(2)]
        for i in range(2):
            pos_scores[i][:, step] = eos_scores[i]
            # convert from cumulative to per-position scores
            pos_scores[i][:, 1:] = pos_scores[i][:, 1:] - pos_scores[i][:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            for i in range(2):
                eos_scores[i] = eos_scores[i] / ((step + 1) ** self.len_penalty)

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # logging.info(f'finished={finished}')
        # logging.info(f'cum_unfin={cum_unfin}')
        # logging.info(f'prev={prev}')
        # logging.info(f'bbsz_idx={bbsz_idx}')

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(min(bbsz_idx[0].size()[0], bbsz_idx[1].size()[0])):
            idx = tuple(bbsz_idx[j][i] for j in range(2))
            score = tuple(eos_scores[j][i] for j in range(2))
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx[0] // beam_size
            # logging.info(f'idx={idx}')
            # logging.info(f'score={score}')
            # logging.info(f'unfin_idx={unfin_idx}')
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # logging.info(f'sent={sent}')
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            # logging.info(f'seen={seen}')
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = tuple([torch.tensor(-math.inf).to(score)]*2)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            # logging.info(f'finished: {finished}')
            if len(finalized[sent]) < beam_size:
                if None not in attn_clone:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent].append(
                    {
                        "tokens": [tokens_clone[j][i] for j in range(2)],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": [pos_scores[j] for j in range(2)],
                    }
                )

        newly_finished: List[int] = []

        # logging.info(f'finished: {finished}')
        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished


class EnsembleModelDD(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)

            # logging.info(f'type(decoder_out)={type(decoder_out)}, len(decoder_out)={len(decoder_out)}')
            # logging.info(f'decoder_out[0]={decoder_out[0]}')
            # logging.info(f'decoder_out[1]={decoder_out[1]}')
            # logging.info(f'len(decoder_out[0])={len(decoder_out[0])}')
            # logging.info(f'decoder_out[0]={[e.shape for e in decoder_out[0]]}')
            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = tuple([attn[i][:, -1, :] for i in range(2)])

            decoder_out_tuple = []
            for i in range(2):
                decoder_out_tuple_tmp = decoder_out[0][i][:, -1:, :].div_(temperature) if decoder_len == 2 else decoder_out[1]
                decoder_out_tuple.append(decoder_out_tuple_tmp)
                # logging.info(f'decoder_out_tuple_tmp: {decoder_out_tuple_tmp}')
                # logging.info(f'decoder_out_tuple_tmp: {decoder_out_tuple_tmp.shape}')
            decoder_out_tuple = (tuple(decoder_out_tuple), None)
            # logging.info(f'len(decoder_out_tuple): {len(decoder_out_tuple)}')
            # logging.info(f'decoder_out_tuple: {[e[0].shape for e in decoder_out_tuple[0]]}')

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            # logging.info(f'probs: {probs}')
            # logging.info(f'probs: {[e.shape for e in probs]}')
            probs = tuple([probs[i][:, -1, :] for i in range(2)])
            if self.models_size == 1:
                # logging.info(f'attn: {attn}')
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )