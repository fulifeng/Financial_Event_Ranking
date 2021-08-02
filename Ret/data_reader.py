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
    def __init__(self, data_file, doc_file, batch_size, shuffle = False):
        print_message("#> Training with the triples in", data_file, "...")
        with open(doc_file,'rb') as f:
            self.documents = pkl.load(f) # list of documents.
        
        with open(data_file,'rb') as f:
            self.triples_raw = pkl.load(f) # ["query", [positive document indexes], [negative document indexes]]
        self.batch_size = batch_size
        self.prepare_triples()
        print_message("loaded", len(self.triples), "data, forming", self.num_ptr, "batch")

    def prepare_triples(self):
        self.triples = []
        for key in self.triples_raw:
            for ind in self.triples_raw[key][0]:
                self.triples.append([key, self.documents[ind], 1])
            negs = self.triples_raw[key][1]
            random.shuffle(negs)
            negs = negs[:len(self.triples_raw[key][0])]
            for ind in negs:
                self.triples.append([key, self.documents[ind], 0])
        random.shuffle(self.triples)
        self.num_ptr = len(self.triples) // self.batch_size
        self.ptr = 0

    def get_minibatch(self):
        if self.ptr == self.num_ptr:
            self.prepare_triples()

        batch = self.triples[self.ptr * self.batch_size: (self.ptr + 1) * self.batch_size]
        self.ptr += 1
        return batch

class DevReader:
    def __init__(self, data_file, doc_file, augment = False):
        print_message("#> Evaluating with the triples in", data_file, "...")
        with open(doc_file,'rb') as f:
            self.documents = pkl.load(f)
        
        with open(data_file,'rb') as f:
            self.triples_raw = pkl.load(f)

        self.augment = augment
        
        self.original_queries = list(self.triples_raw.keys())
        self.triples = [[query, list(set(self.triples_raw[query][0])), list(set(self.triples_raw[query][1]))]  for query in self.triples_raw]
        print_message("Loaded", len(self.triples), "data from", data_file)
        
        if augment:
            self.augment_query()
            print_message("Loaded", len(self.augmented_triples), "data")

    def get_minibatch(self):
        if self.augment:
            for line in self.augmented_triples:
                yield line
        else:
            for line in self.triples:
                yield line

    def augment_query(self):
        self.keys = self.decide_key(self.original_queries[0]) # we make sure that each query corresponds to one keyword.
        self.augmented_triples = []
        for q in self.triples_raw:
            for word in self.keys:
                if word in q:
                    q_target_word = word
                    break
            for key in self.keys:
                self.augmented_triples.append([q.replace(q_target_word, key), list(set(self.triples_raw[q][0])), list(set(self.triples_raw[q][1]))])
                
    def decide_key(self, q):
        candidate_keys = {'chemical':['PTA早','PTA&MEG','橡胶'], 'agriculture':['豆类','棉花','白糖'], 'metal':['有色','黑色','钢矿']}
        for k, v in candidate_keys.items():
            for word in v:
                if word in q:
                    return v
        
        
def save_model_util(model, path):
    if os.path.isdir(path) == False:
        os.mkdir(path)
    print_message("saving model to", path)
    model.save_pretrained(path)
    

def train(args):
    DEVICE = torch.device(args.device)
    
    # building model
    if args.maxsteps==0:
        print_message("Beging Testing...")
        if args.augment:
            print_message("Testing with augmented mode (for training hybrid model)")
        bert = BertForNextSentencePrediction.from_pretrained(args.model_to_test)
        print_message("Load model from", args.model_to_test)
    else:
        print_message("Beging Training...")
        bert = BertForNextSentencePrediction.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    bert = bert.to(DEVICE)

    train_loss = 0.0
    max_eval_loss = 10000

    if args.maxsteps == 0:
        if args.augment: 
            train_reader = DevReader(args.triples_train, args.doc_file, args.augment)
            test_reader = DevReader(args.triples_test, args.doc_file, args.augment)
            dev_reader = DevReader(args.triples_dev, args.doc_file, args.augment)
            
            for name, reader in {'train': train_reader, 'dev': dev_reader, 'test': test_reader}.items():
                result = eval(args.bsize, bert, tokenizer, reader, DEVICE)
                if 'test' not in args.result_save_file:
                    new_filename = args.result_save_file.replace('.json', '_'+name + '.json')
                else:
                    new_filename = args.result_save_file.replace('test', name)
                print_message("Writing hybrid data to file", new_filename)
                with open(new_filename,'w') as f:
                    json.dump(result, f)
        else:
            test_reader = DevReader(args.triples_test, args.doc_file, args.augment)
            _, pred, query, score, _, _ = eval(args.bsize, bert, tokenizer, test_reader, DEVICE)
            with open(args.result_save_file,'w') as f:
                d = {}
                for i in range(len(query)):
                    d[query[i]] = {'pred': pred[i], 'score':score[i]}
                json.dump(d, f)
        return

    reader = TrainReader(args.triples_train, args.doc_file, args.bsize, True)
    eval_reader = DevReader(args.triples_dev, args.doc_file, args.augment)
    test_reader = DevReader(args.triples_test, args.doc_file, args.augment)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(bert.parameters(), lr=args.lr, eps=1e-8)

    optimizer.zero_grad()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.maxsteps)

    for batch_idx in trange(args.maxsteps):
        bert.train()
        Batch = reader.get_minibatch()

        for B_idx in range(args.accumsteps):
            size = args.bsize // args.accumsteps
            B = Batch[B_idx * size: (B_idx+1) * size]
            Q, D, T = zip(*B)

            encoding=tokenizer(Q, D, padding=True, truncation=True, max_length=256)
            input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long, device=DEVICE)
            token_type_ids = torch.tensor(encoding['token_type_ids'], dtype=torch.long, device=DEVICE)
            attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long, device=DEVICE)
            T = torch.tensor(T, dtype=torch.long, device = DEVICE)
            
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
            _, _, _, _, eval_loss, eval_log = eval(args.bsize, bert, tokenizer, eval_reader, DEVICE)
            if eval_loss < max_eval_loss:
                max_eval_loss = eval_loss # recall 10
                print('Max Eval Recall@10 changed to ', max_eval_loss * -1)
                save_model_util(bert, args.model_save_dir)
    
    print("Testing begins...")
    eval(args.bsize, bert, tokenizer, test_reader, DEVICE)


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

def rerank(query, relevant_doc_ids, relevant_docs, model, gt_ids, bsize, tokenizer, DEVICE):
    scores = []
    for D in batch(relevant_docs, bsize):
        encoding = tokenizer([query] * len(D), D, padding=True, truncation=True, max_length=256)
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long, device=DEVICE)
        token_type_ids = torch.tensor(encoding['token_type_ids'], dtype=torch.long, device=DEVICE)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long, device=DEVICE)
        outputs = model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        logits = torch.softmax(outputs[0], dim=1)[:, 1]
        scores.append(logits)

    scores = torch.cat(scores).sort(descending=True)
    ranked = scores.indices.tolist()
    ranked_doc_ids = [relevant_doc_ids[ind] for ind in ranked]
    mrr1 =MRR(gt_ids, ranked_doc_ids, 1)
    mrr3 = MRR(gt_ids, ranked_doc_ids, 3)
    map_ = MAP(gt_ids, ranked_doc_ids)
    recall3 = RECALL(gt_ids, ranked_doc_ids, 3)
    recall5 = RECALL(gt_ids, ranked_doc_ids, 5)
    recall10 = RECALL(gt_ids, ranked_doc_ids, 10)
    #print("%d, MAP: %.6f, MRR1: %.6f, MRR3: %.6f, RECALL3 %.6f, RECALL5 %.6f, RECALL10 %.6f" % (len(ranked_doc_ids), map_, mrr1, mrr3, recall3, recall5, recall10))
    #print('')
    return map_, mrr1, mrr3, recall3, recall5, recall10, ranked_doc_ids[:], scores.values.tolist()[:]

def eval(bsize, model, tokenizer, eval_reader, DEVICE):
    print_message("Evaluation starting...")
    torch.cuda.empty_cache()
    augment = eval_reader.augment
    model.eval()
    maps = []
    mrr1s = []
    mrr3s = []
    recall3s = []
    recall5s = []
    recall10s = []
    save_gt_ids = []
    save_pred_ids = []
    save_queries = []
    save_pred_scores = []
    with torch.no_grad():
        cnt = 1
        if eval_reader.augment:
            target_eval_set = eval_reader.augmented_triples
        else:
            target_eval_set = eval_reader.triples
        for data in tqdm(target_eval_set):  # query pos_ids, neg_ids
            query = data[0]
            save_queries.append(query)
            relevant_doc_ids = data[1] + data[2]
            gt_ids = data[1]
            relevant_docs = [eval_reader.documents[ind] for ind in relevant_doc_ids]
            cnt += 1
            map_, mrr1, mrr3, recall3, recall5, recall10, pred_ids, pred_scores = rerank(query, relevant_doc_ids, relevant_docs, model, gt_ids, bsize, tokenizer, DEVICE)
            save_gt_ids.append(gt_ids)
            save_pred_ids.append(pred_ids)
            save_pred_scores.append(pred_scores)
            maps.append(map_)
            mrr1s.append(mrr1)
            mrr3s.append(mrr3)
            recall3s.append(recall3)
            recall5s.append(recall5)
            recall10s.append(recall10)
            torch.cuda.empty_cache()

    if augment == False:
        # if not augment, print the log
        eval_log = 'Average MAP: %.6f, MRR1: %.6f, MRR3: %.6f, RECALL3: %.6f, RECALL5: %.6f, RECALL10: %.6f' %(np.mean(maps),np.mean(mrr1s),np.mean(mrr3s),np.mean(recall3s),np.mean(recall5s), np.mean(recall10s))
        print(eval_log)
        return save_gt_ids, save_pred_ids, save_queries, save_pred_scores, np.mean(recall10s) * -1, eval_log
    return group_by_query(eval_reader.original_queries, save_queries, save_pred_ids, save_pred_scores)


def group_by_query(original_queries, pred_queries, pred_ids, pred_scores):
    assert len(original_queries) * 3 == len(pred_queries), (len(original_queries), len(pred_queries))
    result_dict = {}
    for i in range(len(original_queries)):
        result_dict[original_queries[i]] = {}
        for j in range(3):
            if pred_queries[i * 3 + j] == original_queries[i]:
                result_dict[original_queries[i]]['pred'] = pred_ids[i * 3 + j]
                result_dict[original_queries[i]]['score'] = [i for i in zip(pred_scores[i * 3 + 0], pred_scores[i * 3 + 1], pred_scores[i * 3 + 2])]
                break
    return result_dict


def batch(group, bsize):
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield L
        offset += len(L)
    return



