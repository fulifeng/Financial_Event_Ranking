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
    def __init__(self, data_file, doc_file, batch_size):
        print_message("#> Train Data reader with the triples in", data_file, "...")
        
        with open(data_file,'rb') as f:
            self.data_raw = pkl.load(f)
        with open(doc_file,'rb') as f:
            self.documents = pkl.load(f)
        
        self.key = self.decide_key(list(self.data_raw.keys())[0])

        self.batch_size = batch_size
        self.prepare_batches()
        self.num_ptr = len(self.data) // self.batch_size
        print_message("loaded", len(self.data), "data, forming", self.num_ptr, "batch")

    def decide_key(self, q):
        candidate_keys = {'chemical':['PTA早','PTA&MEG','橡胶'], 'agriculture':['豆类','棉花','白糖'], 'metal':['有色','黑色','钢矿']}
        for k, v in candidate_keys.items():
            for word in v:
                if word in q:
                    return v

    def prepare_batches(self):
        self.data =[]
        self.ptr = 0
        for q in self.data_raw:
            tag = [0] * 3
            for i in range(3):
                if self.key[i] in q:
                    tag[i] = 1
                    break
            for ind in self.data_raw[q][0]:
                self.data.append([self.documents[ind], tag])
            negs = random.sample(self.data_raw[q][1], len(self.data_raw[q][0]))
            for ind in negs:
                self.data.append([self.documents[ind], [0] * 3])
        random.shuffle(self.data)

    def get_minibatch(self):
        if self.ptr == self.num_ptr:
            self.prepare_batches()
        batch = self.data[self.ptr * self.batch_size: (self.ptr + 1) * self.batch_size]
        self.ptr += 1
        return batch

class DevReader:
    def __init__(self, data_file, doc_file, batch_size):
        print_message("#> Dev Data reader with the triples in", data_file, "...")
        
        with open(data_file,'rb') as f:
            self.data_raw = pkl.load(f)
        with open(doc_file,'rb') as f:
            self.documents = pkl.load(f)
        
        self.key = self.decide_key(list(self.data_raw.keys())[0])
        self.batch_size = batch_size
        self.prepare_batches()
        print_message("Loaded", len(self.data), "data from", data_file)

    def decide_key(self, q):
        candidate_keys = {'chemical':['PTA早','PTA&MEG','橡胶'], 'agriculture':['豆类','棉花','白糖'], 'metal':['有色','黑色','钢矿']}
        for k, v in candidate_keys.items():
            for word in v:
                if word in q:
                    return v

    def prepare_batches(self):
        self.data = []
        for q in self.data_raw:
            for i in range(3):
                if self.key[i] in q:
                    tag = i
                    break
            self.data.append([self.data_raw[q][0], self.data_raw[q][1], tag, q ])
    
    def get_minibatch(self):
        for line in self.data:
            yield line
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
    model.save_pretrained(path)


def train(args):
    DEVICE = torch.device(args.device)

    config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext')
    config.num_labels = 3
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

    train_loss = 0.0
    min_eval_loss = 10000
    
    if args.maxsteps == 0: # test
        print_message("Beging Testing...")
        bert = BertForSequenceClassification.from_pretrained(args.model_to_test, config = config)
        print_message("Load model from", args.model_to_test)
        bert = bert.to(DEVICE)
        test_reader = DevReader(args.triples_test, args.doc_file, args.bsize)
        if args.augment:
            print_message("Testing with augmented mode (for training hybrid model)")
            train_reader = DevReader(args.triples_train, args.doc_file, args.bsize)
            dev_reader = DevReader(args.triples_dev, args.doc_file, args.bsize)
            
            for name, reader in {'train': train_reader, 'dev': dev_reader, 'test': test_reader}.items():
                _, pred_inds, pred_scores, pred_scores_total, queries, _ = test_baseline(bert, tokenizer, reader, args.bsize, DEVICE)
                with open(args.result_save_file.replace('test', name),'w') as f:
                    d = {}
                    for i in range(len(queries)):
                        d[queries[i]] = {'pred':pred_inds[i],'score':pred_scores_total[i], 'target_score': pred_scores[i]}
                    json.dump(d, f)
        else:
            _, pred_inds, pred_scores, _ , queries,_ = test_baseline(bert, tokenizer, test_reader, args.bsize, DEVICE)
            with open(args.result_save_file,'w') as f:
                d = {}
                for i in range(len(queries)):
                    d[queries[i]] = {'pred':pred_inds[i],'score':pred_scores[i]}
                json.dump(d, f)
            
        return
    
    # train
    reader = TrainReader(args.triples_train, args.doc_file, args.bsize)
    eval_reader = DevReader(args.triples_dev, args.doc_file, args.bsize)
    test_reader = DevReader(args.triples_test, args.doc_file, args.bsize)
    bert = BertForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', config = config)
    bert = bert.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(bert.parameters(), lr=args.lr, eps=1e-8)

    optimizer.zero_grad()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.maxsteps)

    for batch_idx in trange(args.maxsteps):
        bert.train()
        Batch = reader.get_minibatch()

        for B_idx in range(args.accumsteps):
            size = args.bsize // args.accumsteps
            B = Batch[B_idx * size: (B_idx+1) * size]
            D, T = zip(*B)

            encoding=tokenizer(D, padding=True, truncation=True, max_length=256)

            input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long, device=DEVICE)
            token_type_ids = torch.tensor(encoding['token_type_ids'], dtype=torch.long, device=DEVICE)
            attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long, device=DEVICE)
            T = torch.tensor(T, dtype = torch.float, device = DEVICE)
            
            outputs = bert(input_ids = input_ids, \
                           token_type_ids = token_type_ids, \
                           attention_mask = attention_mask)

            loss = criterion(outputs[0], T)
            loss = loss / args.accumsteps
            loss.backward()

            train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(bert.parameters(), 2.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (batch_idx + 1) % args.print_log_steps == 0:
            print_message(batch_idx, "Average Train Loss", train_loss / (batch_idx+1))
        if (batch_idx + 1)  % args.do_eval_steps == 0:
            eval_loss, _, _, _, _, eval_log = test_baseline(bert, tokenizer, eval_reader, args.bsize, DEVICE)
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                print('MIN EVAL LOSS changed to %.6f' % min_eval_loss)
                save_model_util(bert, args.model_save_dir)
    print("Testing begins...")
    test_baseline(bert, tokenizer, test_reader, args.bsize, DEVICE)
    return


def test_rerank(target, relevant_doc_ids, relevant_docs, model, gt_ids, bsize, tokenizer, DEVICE):
    scores = []
    total_scores = []
    for D in batch(relevant_docs, bsize):
        encoding = tokenizer(D, padding=True, truncation=True, max_length=256)
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long, device=DEVICE)
        token_type_ids = torch.tensor(encoding['token_type_ids'], dtype=torch.long, device=DEVICE)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long, device=DEVICE)
        outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        logits = torch.sigmoid(outputs[0])
        scores.append(logits[:, target])
        total_scores.append(logits)
    
    scores = torch.cat(scores).sort(descending=True)
    total_scores = torch.cat(total_scores)
    ranked = scores.indices.tolist()
    ranked_doc_ids = [relevant_doc_ids[ind] for ind in ranked]
    total_scores = [total_scores[ind].tolist() for ind in ranked]
    
    mrr1 =MRR(gt_ids, ranked_doc_ids, 1)
    mrr3 = MRR(gt_ids, ranked_doc_ids, 3)
    map_ = MAP(gt_ids, ranked_doc_ids)
    recall3 = RECALL(gt_ids, ranked_doc_ids, 3)
    recall5 = RECALL(gt_ids, ranked_doc_ids, 5)
    recall10 = RECALL(gt_ids, ranked_doc_ids, 10)
    #print(target)
    #print([relevant_docs[ind] for ind in ranked[:3]])
    #print(ranked_doc_ids[:3], gt_ids)
    #print("MAP: %.6f, MRR1: %.6f, MRR3: %.6f, RECALL3 %.6f, RECALL5 %.6f, RECALL10 %.6f" % (map_, mrr1, mrr3, recall3, recall5, recall10))
    #    print('')
    return map_, mrr1, mrr3, recall3, recall5, recall10, ranked_doc_ids, scores.values.tolist(), total_scores



def test_baseline(model, tokenizer, test_reader, bsize, DEVICE):
    torch.cuda.empty_cache()
    model.eval()
    maps = []
    mrr1s = []
    mrr3s = []
    recall3s = []
    recall5s = []
    recall10s = []
    save_pred_inds = []
    save_pred_scores = []
    save_pred_scores_total = []
    save_queries = []
    with torch.no_grad():
        cnt = 1
        for data in tqdm(test_reader.data):  # pos_id, neg_id, tag, q
            relevant_doc_ids = data[0] + data[1]
            random.shuffle(relevant_doc_ids)
            gt_ids = data[0]
            tag = data[2]
            query = data[3]
            save_queries.append(query)
            relevant_docs = [test_reader.documents[ind] for ind in relevant_doc_ids]
            # print(str(cnt) + " Evaluating query --> %s" % query)
            cnt += 1
            map_, mrr1, mrr3, recall3, recall5, recall10, pred_inds, pred_scores, pred_scores_total  = test_rerank(tag, relevant_doc_ids, relevant_docs, model, gt_ids, bsize, tokenizer, DEVICE)
            maps.append(map_)
            mrr1s.append(mrr1)
            mrr3s.append(mrr3)
            recall3s.append(recall3)
            recall5s.append(recall5)
            recall10s.append(recall10)
            save_pred_inds.append(pred_inds)
            save_pred_scores.append(pred_scores)
            save_pred_scores_total.append(pred_scores_total)
            torch.cuda.empty_cache()
    eval_log = 'Average MAP: %.6f, MRR1: %.6f, MRR3: %.6f, RECALL3: %.6f, RECALL5: %.6f, RECALL10: %.6f' %(np.mean(maps),np.mean(mrr1s),np.mean(mrr3s),np.mean(recall3s),np.mean(recall5s), np.mean(recall10s))
    print(eval_log)
    torch.cuda.empty_cache()
    return np.mean(recall10s) * -1, save_pred_inds, save_pred_scores, save_pred_scores_total, save_queries, eval_log

def batch(group, bsize):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield L
        offset += len(L)
    return

