#code by wangchenglong, nlp lab on northeastern university

import math
import torch.nn.functional as F
import collections
import torch
import numpy
import sys
from collections import defaultdict 
from sacrebleu.metrics import BLEU
import sacrebleu

from fairseq.sequence_generator import SequenceGenerator
from fairseq import utils

from fairseq.criterions import LegacyFairseqCriterion, register_criterion, FairseqCriterion

@register_criterion('rl_criterion')
class RLCriterion(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)
        if hasattr(task, 'target_dictionary'):
            tgt_dict = task.target_dictionary
            # self.padding_idx = tgt_dict.split() 
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
            self.eos_idx = self.task.target_dictionary.eos()
        

    @staticmethod
    def add_args(parser): 
        parser.add_argument(
            "--sample-beam",
            default=1,
            type=int,
            metavar="number",
            help="number of sample size",
        )
    
    def forward(self, model, sample, bert_model=None, bert_tokenizer=None, reduce=True):
        #Get hypos
        # sample mode
        #print('!!!RL loss.')
        use_rl_loss = True #use the reinforcement learning to act as loss
        sample = utils.move_to_cuda(sample)

        model.train()

        #NULL Loss
        nsentences = len(sample['id'])
        null_tokens = sample['ntokens']
        net_output = model(**sample['net_input'])

        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = torch.log(probs)
        probs = probs.view(-1, lprobs.size(-1))
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)

        target = target.view(-1, 1)
        null_loss, nll_loss = self.label_smoothed_nll_loss(lprobs, target, ignore_index=self.padding_idx,reduce=True)

        #RL loss
        if use_rl_loss:
            sample_sent_num = 1   # the number of sample
            rl_loss = self.compute_rl_loss(sample['target'], sample_sent_num, net_output[0], self.task.target_dictionary, 
                                            bert_model=bert_model, bert_tokenizer=bert_tokenizer)

        if use_rl_loss:
            # print(rl_loss, null_loss)
            total_loss = rl_loss
            total_loss = rl_loss + null_loss

            total_tokens = null_tokens
        else:
            total_loss = null_loss
            total_tokens = null_tokens
         
        logging_output = {
            'loss': utils.item(total_loss.data),
            'ntokens': total_tokens,
            'sample_size': total_tokens,
            'nsentences':nsentences,
        }
        return total_loss, total_tokens, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
    
    def compute_rl_loss(self, refs, sample_sent_num, probs, tgt_dict,bert_model=None, bert_tokenizer=None):
        # RL loss
        #sample the sents from the pro_dist
        pro_dist = torch.distributions.categorical.Categorical(logits=probs)
        sample_sents = pro_dist.sample([sample_sent_num])     # sample sample_sent_num sentences  [sample_sent_num; snet_len; bzs]
        sent_log_prob = pro_dist.log_prob(sample_sents)

        m,_ = sample_sents[0].size()
        # sample_rewards = torch.zeros([sample_sent_num, m]).float().to(refs.device)

        sample_sents_string = []

        for sample_sent in sample_sents: 
            sample_sents_string += [tgt_dict.string(utils.strip_pad(sent, tgt_dict.pad()).int(), "@@ ") for sent in sample_sent]

        refs_string = [tgt_dict.string(utils.strip_pad(sent, tgt_dict.pad()).int(), "@@ ") for sent in refs]

        #use bleu
        # for i in range(sample_sent_num):
        #     sample_rewards[i] = torch.Tensor(self.get_bleu_score(sample_sents_string[i], refs_string))

        #use bert samilar score
        refs_bert_tokenizer = bert_tokenizer(refs_string, return_tensors="pt", padding=True, truncation=True).to(device=bert_model.device)
        refs_repersentation = bert_model(**refs_bert_tokenizer).pooler_output

        sample_sent_token = bert_tokenizer(sample_sents_string, return_tensors="pt", padding=True, truncation=True).to(device=bert_model.device)
        sample_sents_repersentation = bert_model(**sample_sent_token).pooler_output
        sample_sents_repersentation = sample_sents_repersentation.view(sample_sent_num, -1, len(refs_repersentation[0]))

        refs_repersentation = refs_repersentation.unsqueeze(0)

        sample_rewards = torch.cosine_similarity(sample_sents_repersentation, refs_repersentation, 2)

        #compute the loss
        
        pad_mask = sample_sents.eq(tgt_dict.pad())
        sent_log_prob.masked_fill_(pad_mask, 0.)
        i,j,k = pad_mask.size()
        pad_mask_values = torch.ones([i,j,k]).to(refs.device)
        pad_mask_values.masked_fill_(pad_mask, 0.)

        sent_log_prob = torch.sum(sent_log_prob, axis=-1) / torch.sum(pad_mask_values, axis=-1)  # avg on seq level

        rl_loss = -(torch.nn.functional.relu(sample_rewards-torch.mean(sample_rewards))) * sent_log_prob  # batch_size * 1
        rl_loss = torch.sum(rl_loss)  # avg on batch level 

        return rl_loss

    def label_smoothed_nll_loss(self, lprobs, target, epsilon=0.1, ignore_index=None, reduce=True):
        if target.dim() == lprobs.dim() - 1: 
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        # print(lprobs.size()) [bts*length dict size]  
        # print(target.size()) [bts*length 1]
        # print(nll_loss.size()) [bts*length 1]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def get_bleu_score(self, sys, refs):
        """
        :param sys: sentence 1
        :param refs: sentence 2
        :return: bleu score
        """
        # Note: bleu score 0 if length less than 3
        bleu_scores = [sacrebleu.corpus_bleu([sys[i]], [[refs[i]]]).score for i in range(len(sys))]
        return numpy.array(bleu_scores)
    
    def get_bert_similar_score(self, sys, refs, bertModel=None):
        """
        :param sys: sentence 1
        :param refs: sentence 2
        :return: bert similar score
        """
        