# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import torch.nn.functional as F
import torch.nn as nn

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

    kl_div: bool = field(
        default=False,
        metadata={"help": "whether to use kl divergence loss"},
    )
    
    commit_weight: float = field(
        default=0.25,
        metadata={"help": "weight for commitment loss"},
    )
    img_commit_weight: float = field(
        default=0.25,
        metadata={"help": "weight for image commitment loss"},
    )

    consist_weight: float = field(
        default=1.0,
        metadata={"help": "weight for consist loss"},
    )

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "multimodal_label_smoothed_cross_entropy_match_pretrain", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class MultimodalLabelSmoothedCrossEntropyCriterionMatchPretrain(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        kl_div,
        commit_weight,
        img_commit_weight,
        consist_weight,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.kl_div = kl_div
        self.crossentropyloss = nn.CrossEntropyLoss()
        self.commit_weight = commit_weight
        self.img_commit_weight = img_commit_weight
        self.consist_weight = consist_weight
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),margin=0.1,reduction='mean')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        cor_commit_loss, img_commit_loss, z_q_x_st, z_q_x_st_i = model(**sample["net_input"])
    
        if z_q_x_st is not None:
            img_commit_loss = self.img_commit_weight * img_commit_loss
            sample_size = 1
            consistent_loss = self.consist_weight * (z_q_x_st.mean(0).detach() - z_q_x_st_i.mean(0)).pow(2).mean()
            loss = img_commit_loss + consistent_loss

            logging_output = {
                "loss": loss.data,
                # "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
                "cor_commit_loss": cor_commit_loss.data,
                "img_commit_loss": img_commit_loss.data,
                "consistent_loss":consistent_loss.data,
            }
            

        else:
            logging_output = {
                "loss": loss.data,
                # "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_kl_loss(self, p, q, pad_mask=None):
    
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        if "img_recall" in logging_outputs[0].keys():
            img_loss = sum(log.get("img_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "img_loss", img_loss / sample_size / math.log(2), sample_size, round=3
            )

            img_vq_loss = sum(log.get("img_vq_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "img_vq_loss", img_vq_loss, round=3
            )
            img_commit_loss = sum(log.get("img_commit_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "img_commit_loss", img_commit_loss, round=3
            )
            cor_loss = sum(log.get("cor_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "cor_loss", cor_loss / sample_size / math.log(2), sample_size, round=3
            )
            cor_vq_loss = sum(log.get("cor_vq_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "cor_vq_loss", cor_vq_loss, round=3
            )
            cor_commit_loss = sum(log.get("cor_commit_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "cor_commit_loss", cor_commit_loss, round=3
            )
            
            cor_construct_loss = sum(log.get("cor_construct_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "cor_construct_loss", cor_construct_loss / sample_size / math.log(2), round=3
            )

            consistent_loss = sum(log.get("consistent_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "consistent_loss", consistent_loss / sample_size, round=3
            )



        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
