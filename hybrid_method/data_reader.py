import os
import random
import torch
import torch.nn as nn
import datetime
import json
import pickle as pkl
import numpy as np

from tqdm import *
from argparse import ArgumentParser
from transformers import *

def print_message(*s):
    s = ' '.join([str(x) for x in s])
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

class TrainReader:
    def __init__(self, data_file, doc_file, ret_result, clam_result, batch_size, rerank_num = 100, shuffle = False):
        print_message("#> Data reader with the triples in", data_file, "...")
        with open(data_file,'rb') as f:
            self.data_raw = pkl.load(f)
        with open(doc_file,'rb') as f:
            self.documents = pkl.load(f)
        with open(ret_result, 'r') as f:
            self.r_score = json.load(f)
        with open(clam_result, 'r') as f:
            self.c_score = json.load(f)
            
        self.keys = self.decide_key(list(self.data_raw.keys())[0])
        
        self.data = []
        for q in self.data_raw:
            for i in range(3):
                if self.keys[i] in q:
                    tag = [0] * 3
                    tag[i] = 1
                    break
            assert sum(tag) == 1
            candidate_inds = self.r_score[q]['pred'][:rerank_num] # only rerank the top 100 candidate of ret
            gt_inds = self.data_raw[q][0]
            for ind in candidate_inds:
                if ind in gt_inds:
                    self.data.append([q, ind, tag, tag])
                else:
                    self.data.append([q, ind, [0] * 3, tag])

        self.prepare_score()
        self.batch_size = batch_size
        self.ptr = 0
        self.num_ptr = len(self.data) // self.batch_size

        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.data)

        print_message("loaded", len(self.data), "data, forming", self.num_ptr, "batch")

    def decide_key(self, q):
        candidate_keys = {'chemical':['PTA早','PTA&MEG','橡胶'], 'agriculture':['豆类','棉花','白糖'], 'metal':['有色','黑色','钢矿']}
        for k, v in candidate_keys.items():
            for word in v:
                if word in q:
                    return v
                    
    def prepare_score(self):
        for i, data in enumerate(self.data):
            doc_ind = data[1]
            query = data[0]
            c_score = self.c_score[query]['score'][self.c_score[query]['pred'].index(doc_ind)]
            r_score = self.r_score[query]['score'][self.r_score[query]['pred'].index(doc_ind)]
            self.data[i].append(r_score)
            self.data[i].append(c_score)

    def get_minibatch(self):
        if self.ptr == self.num_ptr:
            if self.shuffle:
                random.shuffle(self.data)
            self.ptr = 0
        batch = self.data[self.ptr * self.batch_size: (self.ptr + 1) * self.batch_size]
        self.ptr += 1
        return batch

    def get_minibatch_dev(self):
        offset = 0
        while offset < len(self.data):
            L = self.data[offset: offset + self.batch_size]
            yield L
            offset += len(L)
        return



def MAP(gt, pred, n=-1):
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(pred):
        if p in gt and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / max(1.0, len(gt))


def RECALL(gt, pred, n = -1):
    if n != -1:
        pred = pred[:n]
    cnt = 0
    for id in pred:
        if id in gt:
            cnt += 1
    return cnt / len(gt)

def MRR(gt, pred, n=-1):
    score = 0.0
    if n == -1:
        n = len(pred)
    for rank, item in enumerate(pred[:n]):
        if item in gt:
            score = 1.0 / (rank + 1.0)
            break
    return score

def save_model_util(model, path):
    if os.path.isdir(path) == False:
        os.mkdir(path)
    print("saving model to %s" % path)
    torch.save({'model': model.state_dict()}, os.path.join(path, './conv_model.pt'))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        # bsize 2 3
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=5, kernel_size=1) # bsize 10 3
        # bsize 3 10
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=5, kernel_size=1) # bsize 10 10
        self.fc = nn.Linear(25, 3)
        self.elu=nn.ELU()
    
    def forward(self, din):
        din = torch.stack((din[:, :3], din[:, 3:]), dim = 1)
        dout = self.conv1(din)
        dout = self.elu(dout)
        dout = dout.permute(0,2,1)
        dout = self.conv2(dout)
        dout = self.elu(dout)
        dout = torch.flatten(dout, start_dim=1)
        dout = torch.sigmoid(self.fc(dout))
        return dout


def train(args):
    DEVICE = torch.device(args.device)

    train_loss = 0.0
    min_eval_loss = 10000
    
    test_reader = TrainReader(args.triples_test, args.doc_file, args.ret_result_file, args.clam_result_file, args.bsize, args.rerank_num, False)
    if args.maxsteps == 0: # test
        print_message("Beging Testing...")
        model = ConvNet()
        ckpt = torch.load(os.path.join(args.model_to_test, 'conv_model.pt'))
        model.load_state_dict(ckpt['model'])
        model = model.to(DEVICE)
        _, result = eval(model, test_reader, DEVICE, None)
        with open(args.result_save_file,'w') as f:
            json.dump(result, f)
        return

    reader = TrainReader(args.triples_train, args.doc_file, args.ret_result_file.replace('test','train'), args.clam_result_file.replace('test','train'), args.bsize, args.rerank_num, True)
    eval_reader = TrainReader(args.triples_dev, args.doc_file, args.ret_result_file.replace('test','dev'), args.clam_result_file.replace('test','dev'), args.bsize, args.rerank_num, False)

    model = ConvNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    
    for batch_idx in trange(args.maxsteps):
        model.train()
        Batch = reader.get_minibatch()

        for B_idx in range(args.accumsteps):
            # q, doc_id, target, r_scores, c_scores
            size = args.bsize // args.accumsteps
            B = Batch[B_idx * size: (B_idx+1) * size]
            D = [line[-2] + line[-1] for line in B]
            tag = [line[2] for line in B]
            q_tag = [line[3] for line in B]
            D = np.array(D)
            tag = np.array(tag)
            q_tag = np.array(q_tag)
            D = torch.cuda.FloatTensor(D)
            tag = torch.cuda.FloatTensor(tag)
            q_tag = torch.cuda.FloatTensor(q_tag)
            output = model(D) # bsize 3
            output = output *  q_tag
            loss = criterion(output, tag)
            loss = loss / args.accumsteps
            loss.backward()
            
            train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx + 1) % args.print_log_steps == 0:
            print_message(batch_idx, train_loss / (batch_idx+1))
        if (batch_idx + 1)  % args.do_eval_step == 0:
            eval_loss, _ = eval(model, eval_reader, DEVICE, criterion)
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                print_message("MIN EVAL LOSS changed to", min_eval_loss)
                save_model_util(model, args.model_save_dir)
    print_message("Testing begins...")
    eval(model, test_reader, DEVICE, None)
    return


def calc_scores(result, data_raw):
    maps = []
    mrr1s = []
    mrr3s = []
    recall3s = []
    recall5s = []
    recall10s = []
    for q in result:
        new_ranked = [(doc_id,score) for doc_id, score in result[q].items()]
        new_ranked = sorted(new_ranked, key=lambda x: x[1])[::-1]
        new_ranked = [line[0] for line in new_ranked]
        maps.append(MAP(data_raw[q][0], new_ranked))
        mrr1s.append(MRR(data_raw[q][0], new_ranked, 1))
        mrr3s.append(MRR(data_raw[q][0], new_ranked, 3))
        recall3s.append(RECALL(data_raw[q][0], new_ranked, 3))
        recall5s.append(RECALL(data_raw[q][0], new_ranked, 5))
        recall10s.append(RECALL(data_raw[q][0], new_ranked, 10))
    # print(maps[-1], mrr1s[-1], mrr3s[-1], recall3s[-1], recall5s[-1], recall10s[-1])
    print('Average MAP: %.6f, MRR1: %.6f, MRR3: %.6f, RECALL3: %.6f, RECALL5: %.6f, RECALL10: %.6f' %(np.mean(maps),np.mean(mrr1s),np.mean(mrr3s),np.mean(recall3s),np.mean(recall5s), np.mean(recall10s)))
    return np.mean(maps),np.mean(mrr1s),np.mean(mrr3s),np.mean(recall3s),np.mean(recall5s), np.mean(recall10s)


def eval(model, eval_reader, DEVICE, criterion):
    print("Evaluation starting...")
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        cnt = 0
        result = {}
        eval_loss = 0
        for B in eval_reader.get_minibatch_dev():  # d, t
            cnt += 1
            # q, doc_id, target, q_tag, r_scores, c_scores
            D = [line[-2] + line[-1] for line in B]
            tag = [line[2] for line in B]
            q_tag = [line[3] for line in B]
            D = np.array(D)
            tag = np.array(tag)
            q_tag = np.array(q_tag)
            D = torch.cuda.FloatTensor(D)
            tag = torch.cuda.FloatTensor(tag)
            q_tag = torch.cuda.FloatTensor(q_tag)
            output = model(D)
            output = output *  q_tag
            if criterion is not None:
                loss = criterion(output, tag)
                eval_loss += loss.item()
            output = torch.sum(output, dim=-1).cpu().tolist()
            for i in range(len(B)):
                assert output[i] > 0
                if B[i][0] not in result:
                    result[B[i][0]] = { B[i][1] : output[i] }
                else:
                    result[B[i][0]][B[i][1]] = output[i]
        eval_loss = eval_loss / cnt
        print('Evaluation loss', eval_loss)
        map_, mrr1, mrr3, recall3, recall5, recall10 = calc_scores(result, eval_reader.data_raw)
    return recall10 * -1, result




