# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from os import device_encoding
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.sequence_generator import SequenceGenerator
import queue

#use MLP model
class Q_Modle(nn.Module):
    def __init__(self, hidden_size, voc_class):
        super(Q_Modle, self).__init__()
        self.input_projection = nn.Linear(hidden_size, 1024, bias=True)
        self.output_projection = nn.Linear(1024, voc_class, bias=True)
    
    def forward(self, X):
        output =  self.output_projection(self.input_projection(X))
        output = output.sigmoid()
        return output

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1: 
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
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


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.range_eps = 0.01
        self.queue = torch.cuda.FloatTensor([])
        self.teacher_loss_queue =  torch.cuda.FloatTensor([])
        self.real_distil_rate = 0.0
        self.dict_count = None
        self.sample_res = []
        self.inner_res = []
        self.targets = []
        self.cbatch = 0
        self.tar_dict = task.target_dictionary
        self.split = self.tar_dict.index("@@@")
        # print("<split>'s index is:", self.split)

    @staticmethod 
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on
    
    def push_to_FIFO_queue(self, tensor):
        tensor = tensor.detach().view(-1)
        tensor_size = tensor.size(0)
        current_size = self.queue.size(0)
        self.queue = self.queue.view(-1)
        if tensor_size + current_size < self.task.args.difficult_queue_size:
            self.queue = torch.cat((self.queue, tensor))
        else:
            self.queue = torch.cat((self.queue[tensor_size: ], tensor))
    
    def push_to_teacher_FIFO_queue(self, tensor):
        tensor = tensor.detach().view(-1)
        tensor_size = tensor.size(0)
        current_size = self.teacher_loss_queue.size(0)
        self.teacher_loss_queue = self.teacher_loss_queue.view(-1)
        if tensor_size + current_size < self.task.args.difficult_queue_size:
            self.teacher_loss_queue = torch.cat((self.teacher_loss_queue, tensor))
        else:
            self.teacher_loss_queue = torch.cat((self.teacher_loss_queue[tensor_size: ], tensor))

    def forward(self, model, sample, reduce=True, teacher_model=None, update_num=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])   # 1: [len, btz, hidden size]

        teacher_output = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(**sample['net_input'])
        
        #save the inner information and beam information
        # sample_beam = 3
        # translator = SequenceGenerator([model], tgt_dict=self.tar_dict, beam_size=sample_beam)
        # hypos = translator.generate(model, sample)

        # for batch_id, sample_batch in enumerate(hypos):
        #     for beam_id, sample_beam in enumerate(sample_batch):
        #         hypos[batch_id][beam_id] =  hypos[batch_id][beam_id]['tokens'].tolist()

        # self.sample_res.append(hypos)
        # self.inner_res.append(net_output[1]['inner_states'][-1].detach())
        # self.targets.append(sample['target'])
        # self.cbatch += 1
        # print(self.cbatch)
        # if self.cbatch%100 == 0:
        #     save_id = self.cbatch//100
        #     np.save(f'/home/wangchenglong/deep_rl/q_function/data/sample_res_{save_id}.npy', self.sample_res)
        #     np.save(f'/home/wangchenglong/deep_rl/q_function/data/inner_res_{save_id}.npy', self.inner_res)
        #     np.save(f'/home/wangchenglong/deep_rl/q_function/data/targets_{save_id}.npy', self.targets)
        #     self.sample_res = []
        #     self.inner_res = []
        #     self.targets = []

        
        #use the q_function to train NMT model
        class Q_Modle(nn.Module):
            def __init__(self, hidden_size, voc_class):
                super(Q_Modle, self).__init__()
                self.input_projection = nn.Linear(hidden_size, 1024, bias=True)
                self.output_projection = nn.Linear(1024, voc_class, bias=True)
            
            def forward(self, X):
                output =  self.output_projection(self.input_projection(X))
                output = output.sigmoid()
                return output
                
        use_q_function = True
        if use_q_function:
            voc_class = len(self.tar_dict)
            hidden_size = 512
            q_model_path = '/home/wangchenglong/deep_rl/q_function/models/q_model_11.pkl'
            q_model_arguments = torch.load(q_model_path)
            q_model = Q_Modle(hidden_size, voc_class)
            q_model.load_state_dict(q_model_arguments)
            q_model.to(net_output[0].device)
            q_model.eval()
            loss, nll_loss, extra_result = self.compute_loss(model, net_output, sample, reduce=reduce, 
                                                teacher_output=teacher_output, 
                                                distil_strategy=self.task.args.distil_strategy,
                                                update_num=update_num,
                                                use_q_function=True,
                                                q_model=q_model)
        else:
            loss, nll_loss, extra_result = self.compute_loss(model, net_output, sample, reduce=reduce, 
                                                teacher_output=teacher_output, 
                                                distil_strategy=self.task.args.distil_strategy,
                                                update_num=update_num)

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data if nll_loss is not None else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'distil_rate': self.real_distil_rate,
            'gpu_nums':1,
            'KD_loss': extra_result['KD_loss'].data if extra_result.get('KD_loss', None) is not None else 0,  
            'nll_loss_distil': extra_result['nll_loss_distil'].data if extra_result.get('nll_loss_distil', None) is not None else 0,  
            #'distil_token_num': extra_result['distil_token_num'].data if extra_result.get('distil_token_num', None) is not None else 0,  
        }
        
        return loss, sample_size, logging_output

    def get_teacher_probs(self, teacher_output):
        teacher_predict = teacher_output[0]
        teacher_predict = teacher_predict.view(-1, teacher_predict.size(-1)) # B*T x vocab
        if self.task.args.teacher_predict_temperature_schedule == 'binary':
            teacher_predict_max = torch.max(teacher_predict, dim=-1)[0].view(-1, 1) # B*T x 1
            teacher_predict_mask = teacher_predict_max > 0.5 # B*T x 1
            temperature = torch.ones_like(teacher_predict_max) / self.task.args.teacher_predict_temperature # B*T x 1 
            temperature = temperature.masked_fill(teacher_predict_mask, self.task.args.teacher_predict_temperature) # B*T x 1
            teacher_predict = teacher_predict * temperature
        elif self.task.args.teacher_predict_temperature_schedule == 'topk':
            distil_lprobs = F.softmax(teacher_predict, dim=-1, dtype=torch.float32) # B * T x vocab
            distil_mask = distil_lprobs > 0.01
            invalid_mask = distil_mask.sum(dim=-1) == 0
            distil_mask[invalid_mask, :] = True
            teacher_predict.masked_fill_(~distil_mask, float("-inf"))
        else:
            teacher_predict = teacher_predict * self.task.args.teacher_predict_temperature
        distil_lprobs = F.softmax(teacher_predict, dim=-1, dtype=torch.float32) # B x T x vocab

        return distil_lprobs

    def compute_loss(self, 
        model, 
        net_output, 
        sample, 
        reduce=True, 
        teacher_output=None, 
        distil_strategy="normal", 
        update_num=None,
        use_q_function=False,
        q_model=None):

        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = torch.log(probs)
        probs = probs.view(-1, lprobs.size(-1))
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)

        bsz, seq_len = target.shape
        target = target.view(-1, 1)
        pad_mask = target.eq(self.padding_idx).view(-1)
        loss = None
        nll_loss = None
        extra_result = {}

        if use_q_function:
            #use the q function to train nmt model
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )

            model_inners = net_output[1]['inner_states'][-1]
            q_dist = q_model(model_inners.to(torch.float32))
            q_dist = q_dist.view(-1, q_dist.size(-1)).to(torch.float16)
            q_loss = -q_dist*lprobs
            q_loss = q_loss.sum(-1)
            q_loss.masked_fill_(pad_mask, 0.)
            q_loss = q_loss.mean()
            loss = loss + q_loss
        else:
            # not use q_function to train nmt model
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )

        return loss, nll_loss, extra_result

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        #kd_loss_sum = sum(log.get('KD_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nll_loss_distil = sum(log.get('nll_loss_distil', 0) for log in logging_outputs)
        distil_token_num = sum(log.get('distil_token_num', 0) for log in logging_outputs)
        GPU_nums = sum(log.get('gpu_nums', 0) for log in logging_outputs)
        real_distil_rate = sum(log.get('distil_rate', 0) for log in logging_outputs) / GPU_nums
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        #metrics.log_scalar('kd_loss_sum', kd_loss_sum / distil_token_num, round=4)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('distil_rate', real_distil_rate, round=4)
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
