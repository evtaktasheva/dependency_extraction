import json
import numpy as np
import os
import pandas as pd
import random
import torch
from conllu import parse
from tqdm.auto import tqdm
from udapi.block.read.conllu import Conllu
from io import StringIO
from collections import defaultdict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def map_seq_bert(tokens, ud_sent):
    merged = []
    mapped = []
    buf = []
    ud = 0
    i = 0
    for t in range(len(tokens)):
        if tokens[t] in ['[CLS]', '[SEP]']:
            mapped.append(-1)
        elif tokens[t].startswith('#'):
            buf.append(tokens[t][2:])
            mapped.append(mapped[t-1])
        else:
            if ''.join(buf) == ud_sent[ud]:
                merged.append(''.join(buf))
                ud += 1
                i += 1
                buf = [tokens[t]]
                mapped.append(i)
            else:
                if buf == ['[UNK]']:
                    i += 1
                    mapped.append(i)
                    buf = [tokens[t]]
                elif t == 1:
                    mapped.append(i)
                else:
                    mapped.append(mapped[t-1])
                buf.append(tokens[t])
    return mapped
    
    
def map_seq_xlmr(tokens, ud_sent):
    merged = []
    mapped = []
    buf = []
    ud = 0
    i = 0
    for t in range(len(tokens)):
        if tokens[t] in ['<s>', '</s>']:
            mapped.append(-1)
        elif not tokens[t].startswith('▁'):
            buf.append(tokens[t])
            mapped.append(mapped[t-1])
        else:
            if ''.join(buf) == ud_sent[ud]:
                merged.append(''.join(buf))
                ud += 1
                i += 1
                buf = [tokens[t].strip('▁')]
                mapped.append(i)
            else:
                if buf == ['<unk>']:
                    i += 1
                    mapped.append(i)
                    buf = [tokens[t].strip('▁')]
                elif t == 1 or t == 0:
                    mapped.append(i)
                else:
                    mapped.append(mapped[t-1])
                buf.append(tokens[t].strip('▁'))
    return mapped


def get_subwords(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def get_ud_analysis(analysis):
    analysis = Conllu(filehandle=StringIO(analysis)).read_tree()
    heads = []
    words = [node.form for node in analysis.descendants]
    for id, token in enumerate(analysis.descendants):
        if token.deprel == 'root':
            # root_id = words.index(token.form)
            heads.append(-1)
        else:
            heads.append(words.index(token.parent.form))
    return heads, words
    

def get_deprels(heads, tokens):
    deprels = []
    tokens = [token.lower() for token in tokens]
    for i, head in enumerate(heads):
        deprels.append((tokens[i], tokens[head]))
    return deprels

   
def merge_tokens(tokens, mapping):
    merged = []
    buf = []
    for t in range(len(mapping)):
        if t == 0:
            buf.append(tokens[t].strip('▁'))
        elif mapping[t] == mapping[t-1]:
            buf.append(tokens[t].strip('#').strip('▁'))
        else:
            merged.append(''.join(buf))
            buf = [tokens[t].strip('▁')]
    merged.append(''.join(buf))
    return merged


def uas(y_pred, y_true):
    return len(set(y_pred) & set(y_true)) / len(y_true) 


def uuas(y_pred, y_true):
    y_pred = y_pred + [y[::-1] for y in y_pred]
    return len(set(y_pred) & set(y_true)) / len(y_true) 
  
  
def get_true_values(trees, layers=12):
    trees = [[tree]*layers for tree in trees]
    out = []
    for layer in trees:
        out.extend(layer)
    return(out)


def iou(y_pred, y_true):
    y_pred = set(y_pred + [y[::-1] for y in y_pred])
    y_true = set(y_true + [y[::-1] for y in y_true])
    return len(y_pred.intersection(y_true)) / len(y_pred.union(y_true))
   

def create_results_directory(save_dir_name: str, task: str, model: str, probe=None) -> str:
    model = model.replace('/', '-')
    if probe:
        probe_task_dir_path = os.path.join(os.getcwd(), 'results', save_dir_name, task, model, probe)
    else:
        probe_task_dir_path = os.path.join(os.getcwd(), 'results', save_dir_name, task, model) 
    if not os.path.exists(probe_task_dir_path):
        os.makedirs(probe_task_dir_path)

    return probe_task_dir_path


def save_results(prober: str, model: str, task: str, position_embedding: str, data: dict):
    dir_path = create_results_directory(task=task, model=model, save_dir_name='metrics')
    file = f'{prober}_{position_embedding}_results.json'
    path = os.path.join(dir_path, file)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def sort_matrix(a, sent):
    rows = []
    # sort columns
    for row in a:
        row = [i[0] for i in sorted(zip(row, sent), key=lambda x: x[1])]
        rows.append(row)
    # sort rows
    cols = [i[0] for i in sorted(zip(rows, sent), key=lambda x: x[1])]
    return np.array(cols)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=-1):
        self.labels = [str(label) for label in df.label]
        self.correct_sents = [sent for sent in df.correct_sent]
        self.incorrect_sents = [sent for sent in df.incorrect_sent]
        self.cor_roots = [int(root) for root in df.root]
        self.inc_roots = [int(root) for root in df.inc_root]
        heads = []
        ud_sents = []
        for sent in df.annotation:
            head, ud_sent = get_ud_analysis(sent) 
            heads.append(head)
            ud_sents.append(ud_sent)
        self.deprels = [get_deprels(heads[i], ud_sents[i]) for i in range(len(df))]

        if not max_len == -1:
            print(f'Shortening to {max_len}')
            self.labels = self.labels[:max_len]
            self.correct_sents = self.correct_sents[:max_len]
            self.incorrect_sents = self.incorrect_sents[:max_len]
            self.cor_roots = self.cor_roots[:max_len]
            self.inc_roots = self.inc_roots[:max_len]
            self.deprels = self.deprels[:max_len]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.labels[idx], self.correct_sents[idx], self.incorrect_sents[idx], 
                self.cor_roots[idx], self.inc_roots[idx], self.deprels[idx])


class DataLoader(object):
    def __init__(self, filename, max_len=-1):
        self.filename = filename
        self.max_len = max_len
        self.path = os.path.join(os.getcwd(), 'data')

    def load_data(self):
        data = pd.read_csv(os.path.join(self.path, self.filename + '.txt'), delimiter='\t')
        return Dataset(data, max_len=self.max_len)
