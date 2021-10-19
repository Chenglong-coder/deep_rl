from os import truncate
import numpy as np
from numpy.lib.function_base import diff
import torch
import torch.nn as nn
import torch.optim as optim
from sacrebleu.metrics import BLEU
import sacrebleu
from fairseq.data import Dictionary
from transformers import BertTokenizer, BertModel

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

class Trainer_Q(nn.Module):
    def __init__(self, tar_dict, bert_tokenizer=None, bert_model=None, loss_fn=None):
        super(Trainer_Q, self).__init__()
        self.tar_dict = tar_dict
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.loss_fn = loss_fn
        self.eos_id = self.tar_dict.eos()
 
    def forward(self, sample_batch, inner_batch, targets_batch, ref_model, train_model, optimizer, device):
        inner_batch = torch.transpose(inner_batch, 0, 1)
        if self.bert_model != None:
            bert_model_device = "cpu"
            self.bert_model.to(torch.device(bert_model_device))

        for batch_id, sample in enumerate(sample_batch):
            optimizer.zero_grad()

            inner = inner_batch[batch_id]
            q_distribution_ref = ref_model(inner)[:-1]
            q_distribution = train_model(inner)[1:]

            dist = q_distribution - q_distribution_ref

            tar = targets_batch[batch_id].to(torch.int32)
            sample = torch.Tensor(sample[0]).to(torch.int32)

            all_sample = torch.ones(len(sample)-1, len(sample)-1).to(torch.int32)*self.eos_id
            for line_id in range(len(sample)-1):
                all_sample[line_id][0:line_id+1] = sample[0:line_id+1]
                
            #using the bert model
            tar_string = self.tar_dict.string(tar, bpe_symbol='@@ ')
            sample_strings = [self.tar_dict.string(sample_, bpe_symbol='@@ ') for sample_ in all_sample]
            del all_sample

            #using the finetune to compute score
            # refs_bert_tokenizer = self.bert_tokenizer(tar_string, return_tensors="pt", \
            #     padding=True, truncation=True).to(torch.device(bert_model_device))
            # refs_repersentation = self.bert_model(**refs_bert_tokenizer).pooler_output.to(device)

            # sample_bert_tokenizer = self.bert_tokenizer(sample_strings, return_tensors="pt", \
            #     padding=True, truncation=True).to(torch.device(bert_model_device))
            # sample_repersentation = self.bert_model(**sample_bert_tokenizer).pooler_output.to(device)

            # sample_repersentation = sample_repersentation.view(len(sample)-1, -1, len(refs_repersentation[0]))
            # refs_repersentation = refs_repersentation.unsqueeze(0)
            # sample_rewards = torch.cosine_similarity(sample_repersentation, refs_repersentation, 2)

            #using the bleu
            sample_rewards = torch.Tensor(get_bleu_score(sample_strings, tar_string))
            
            #get the true reward
            true_reward = torch.Tensor(sample_rewards.size(0)-1).to(device)
            for index in range(true_reward.size(0)):
                true_reward[index] = sample_rewards[index+1] - sample_rewards[index]

            predict = torch.Tensor(true_reward.size(0)).to(device)
            
            for word_id in range(true_reward.size(0)):
                if word_id >= dist.size(0):
                    predict[word_id] = true_reward[word_id]
                else:
                    predict[word_id] = dist[word_id][sample[word_id]]

            loss = loss_fn(predict, true_reward)
            
            loss.backward()
            optimizer.step()
        print(loss)

def get_bleu_score(sys, ref):
    """
    :param sys: sentence 1
    :param refs: sentence 2
    :return: bleu score
    """
    # Note: bleu score 0 if length less than 3
    bleu_scores = [sacrebleu.corpus_bleu([sys[i]], [[ref]]).score*0.01 for i in range(len(sys))]
    return np.array(bleu_scores)

if __name__ == '__main__':
    inner_res = []
    sample_res = []
    refs_res = []
    device  = torch.device(0)
    #voc
    tar_dict = Dictionary()
    tar_dict.add_from_file('/home/wangchenglong/deep_rl/data-bin/iwslt14de2en/dict.en.txt')
    voc_class = len(tar_dict)
    hidden_size = 512

    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    trainer = Trainer_Q(tar_dict, loss_fn=loss_fn) 

    dir_path = '/home/wangchenglong/deep_rl/q_function/data/'
    for file_index in range(1,12):
        inner_data_path = dir_path+f'inner_res_{file_index}.npy'
        sample_data_path = dir_path+f'sample_res_{file_index}.npy'
        refs_data_path = dir_path+f'targets_{file_index}.npy'

        inner_data = np.load(inner_data_path, allow_pickle=True)
        sample_data = np.load(sample_data_path, allow_pickle=True)
        refs_data = np.load(refs_data_path, allow_pickle=True)

        inner_res.append(inner_data)
        sample_res.append(sample_data)
        refs_res.append(refs_data)

    ref_model = Q_Modle(hidden_size, voc_class)
    train_model = Q_Modle(hidden_size, voc_class)
    counter = 1
    for sample_beams, inners, refs in zip(sample_res, inner_res, refs_res):     #inner [sent len; batch size; hidden size]
        ref_model.eval()
        ref_model.to(device)
        train_model.train()
        train_model.to(device)
        q_optimizer = optim.Adam(train_model.parameters(), lr=0.001)

        for sample_beam, inner, ref in zip(sample_beams, inners, refs):
            trainer(sample_beam, inner.to(torch.float32).to(device), ref, ref_model, train_model, q_optimizer, device)
        
        torch.save(train_model, f"models/q_model_{counter}.pkl")
        ref_model = torch.load(f"models/q_model_{counter}.pkl")

        counter += 1
        