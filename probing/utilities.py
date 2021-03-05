import json
import numpy as np
import os
import pandas as pd
import random
import torch
from conllu import parse


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def map_seq(tokens):
    mapped = []
    for t, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            mapped.append(-1)
        elif token.startswith('##'):
            mapped.append(mapped[t - 1])
        else:
            if t == 0:
                mapped.append(t)
            else:
                mapped.append(mapped[t - 1] + 1)
    return mapped


def get_subwords(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
    return id_for_all_subwords


def perturbe_ud(root_id, ud_heads):
    for i, token in enumerate(ud_heads):
        if token == 0:
            ud_heads[i] = root_id
        elif token == -1:
            ud_heads[0], ud_heads[i] = token, ud_heads[0]
        elif token == root_id:
            ud_heads[i] = 0
    return ud_heads


def get_ud_analysis(sentence, pipeline):
    analysis = parse(pipeline.process(sentence))
    heads = []
    for token in analysis[0]:
        if token['deprel'] == 'root':
            root = token['id'] - 1
            heads.append(-1)
        else:
            heads.append(token['head'] - 1)
    return root, perturbe_ud(root, heads)


def merge_tokens(tokens):
    out = []
    buf = []
    for token in tokens:
        if token.startswith('##'):
            buf.append(token[2:])
        else:
            if len(buf) > 1:
                out.append(''.join(buf))
                buf = [token]
            else:
                out.extend(buf)
                buf = [token]
    out.extend(buf)
    return out


def get_deprels(heads, tokens):
    deprels = []
    for i, head in enumerate(heads):
        deprels.append((tokens[i], tokens[head]))
    return deprels


def uas(y_pred, y_true):
    y_pred.sort()
    y_true.sort()
    count = 0
    for i, deprel in enumerate(y_pred):
        if deprel == y_true[i]:
            count += 1
    return count / len(y_pred)


def uuas(y_pred, y_true):
    y_pred.sort()
    y_true.sort()
    count = 0
    for i, deprel in enumerate(y_pred):
        if deprel == y_true[i] or deprel[::-1] == y_true[i]:
            count += 1
    return count / len(y_pred)


def create_results_directory(task: str, save_dir_name: str) -> str:
    probe_task_dir_path = os.path.join(os.getcwd(), save_dir_name, task)

    if not os.path.exists(probe_task_dir_path):
        os.makedirs(probe_task_dir_path)

    return probe_task_dir_path


def save_results(task, data):
    dir_path = create_results_directory(task, save_dir_name='results')
    file = f'{task}_results.json'
    path = os.path.join(dir_path, file)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=-1):
        self.sentences = [sent for sent in df.sentence]
        self.tokenized = [tokenizer.tokenize(sent) for sent in df.sentence]
        self.labels = [label for label in df.label]
        self.deprels = [deprel for deprel in df.deprels]
        self.roots = [int(root) for root in df.root_id]

        if not max_len == -1:
            print(f'shortening to {max_len}')
            self.sentences = self.sentences[:max_len]
            self.tokenized = self.tokenized[:max_len]
            self.labels = self.labels[:max_len]
            self.deprels = self.deprels[:max_len]
            self.roots = self.roots[:max_len]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tokenized[idx], self.labels[idx], self.deprels[idx], self.roots[idx]


class DataLoader(object):
    def __init__(self, filename, tokenizer, max_len=-1):
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.path = os.path.join(os.getcwd(), 'data')

    def load_data(self):
        data = pd.read_csv(os.path.join(self.path, self.filename + '.txt'), delimiter='\t')
        return Dataset(data, self.tokenizer, max_len=self.max_len)
