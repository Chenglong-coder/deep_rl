#code by wangchenglong, nlp lab on northeastern university

import math
import torch.nn.functional as F
import collections
import torch
import numpy
import sys
from collections import defaultdict 
from sacrebleu.metrics import BLEU

from fairseq.sequence_generator import SequenceGenerator
from fairseq import utils, libbleu

from fairseq.criterions import LegacyFairseqCriterion, register_criterion, FairseqCriterion

@register_criterion('rl_criterion_back')
class RLCriterion_Back(FairseqCriterion):

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
    
    def forward(self, model, sample, reduce=True):
        #Get hypos
        # sample mode
        #print('!!!RL loss.')
        use_rl_loss = False #use the reinforcement learning to act as loss

        if use_rl_loss:
            model.eval()
            # src_dict = self.task.source_dictionary
            tgt_dict = self.task.target_dictionary
            eos_idx = self.task.target_dictionary.eos()
            sample_beam = 3

            translator = SequenceGenerator([model], tgt_dict=tgt_dict, beam_size=sample_beam, min_len=1)

            translator.to(sample['id'].device)

            s = utils.move_to_cuda(sample)

            with torch.no_grad():
                hypos = translator.generate(model, s)  #[batch size; beam sample size; dict(['tokens', 'score', 'attention', 'alignment', 'positional_scores']))]

            for i in range(len(hypos)):
                for j in range(len(hypos[i])):
                    for del_key_name in ['score', 'attention', 'alignment', 'positional_scores']:
                        del hypos[i][j][del_key_name]

        model.train()

        #NULL Loss
        nsentences = len(sample['id'])
        null_tokens = sample['ntokens']
        net_output = model(**sample['net_input']) 

        print("&&&&", net_output[0].size())

        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = torch.log(probs)
        probs = probs.view(-1, lprobs.size(-1))
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)

        target = target.view(-1, 1)
        null_loss, nll_loss = self.label_smoothed_nll_loss(lprobs, target, ignore_index=self.padding_idx,reduce=True)

        if use_rl_loss:
            rl_loss = self.compute_rl_loss(s['target'], s['net_input']['src_tokens'], model, hypos, sample_beam, tgt_dict, self.eos_idx)
            total_loss = 0.5 * rl_loss + 0.5 * null_loss
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
        # print('total: ',total_loss)
        # print("******************", total_loss, total_tokens)
        # exit(0)
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
    
    def compute_rl_loss(self, refs, srcs, model, hypos, sample_beam, tgt_dict, eos_idx):
        # RL loss
        rl_loss = 0
        for ref, src, hypo in zip(refs, srcs, hypos):
            # print(ref)
            # print(hypo)
            target_tokens = utils.strip_pad(ref, tgt_dict.pad()).int()
            ref = tgt_dict.string(target_tokens, "@@ ")
            
            rewards = torch.Tensor(sample_beam).float().to(refs.device)
            logprobs = torch.Tensor(sample_beam).float().to(refs.device)
            for i in range(sample_beam):
                trans_id = hypo[i]['tokens']
                hypo_tokens = utils.strip_pad(trans_id, tgt_dict.pad()).int()
                trans_str = tgt_dict.string(hypo_tokens, "@@ ")
                rewards[i] = float(str(bleu.corpus_score([ref], [trans_str]))[7:12])
                
                tgt_input_tokens = trans_id.new(trans_id.shape).fill_(0)
                assert trans_id[-1] == eos_idx
                tgt_input_tokens[0] = eos_idx
                tgt_input_tokens[1:] = trans_id[:-1]

                train_sample = {
                    'net_input': {
                        'src_tokens': src.view(1, -1),
                        'src_lengths': torch.LongTensor(src.numel()).view(1, -1),
                        'prev_output_tokens': tgt_input_tokens.view(1, -1),
                    },
                    'target': trans_id.view(1, -1)
                }

                train_sample = utils.move_to_cuda(train_sample)
                net_output = model(**train_sample['net_input'])
                lprobs = model.get_normalized_probs(net_output, log_probs=True)
                lprobs = lprobs.view(-1, lprobs.size(-1))
                target = model.get_targets(train_sample, net_output).view(-1, 1)
                non_pad_mask = target.ne(tgt_dict.pad())
                lprob = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
                logprobs[i] = torch.sum(lprob)
            rl_loss += torch.sum(logprobs*(rewards))
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