import warnings

warnings.simplefilter(action="ignore", category=(FutureWarning, DeprecationWarning))

import json
import numpy as np
import torch
from tqdm.auto import tqdm
from ufal.chu_liu_edmonds import chu_liu_edmonds
from probing.utilities import *


class ExtractDependencies(object):
    def __init__(self, data, model, tokenizer, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        if self.args.cuda:
            self.model.to('cuda')

        self.model = self.model.eval()

    def get_dep_matrix(self):
        LAYER = int(self.args.layers)
        LAYER += 1  # embedding layer
        out = [[] for _ in range(LAYER)]

        # generate masks
        # get id for [MASK]
        mask_id = self.tokenizer.encode('[MASK]')[1]

        for ind, (sentence, tokens, label, deprels, root_id) in tqdm(enumerate(self.data), total=len(self.data)):

            # Convert token to vocabulary indices
            tokenized_text = list(tokens)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            # map bpe tokens to words
            mapping = map_seq(tokenized_text)

            # generating mask indices
            all_layers_matrix_as_list = [[] for i in range(LAYER)]

            l_tokens = len(tokenized_text)

            for i in range(0, l_tokens):

                id_for_all_i_tokens = get_subwords(mapping, i)
                tmp_indexed_tokens = list(indexed_tokens)
                # mask all bpe tokens of a word
                for tmp_id in id_for_all_i_tokens:
                    tmp_indexed_tokens[tmp_id] = mask_id
                one_batch = [list(tmp_indexed_tokens) for _ in range(l_tokens)]

                for j in range(l_tokens):
                    id_for_all_j_tokens = get_subwords(mapping, j)
                    # mask all bpe tokens of a word
                    for tmp_id in id_for_all_j_tokens:
                        one_batch[j][tmp_id] = mask_id

                tokens_tensor = torch.tensor(one_batch)
                segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
                if self.args.cuda:
                    tokens_tensor = tokens_tensor.to('cuda')
                    segments_tensor = segments_tensor.to('cuda')

                # get hidden states
                with torch.no_grad():
                    model_outputs = self.model(tokens_tensor, segments_tensor)
                    all_layers = model_outputs.hidden_states  # 12 layers + embedding layer

                # get hidden states for word_i
                for k, layer in enumerate(all_layers):
                    if self.args.cuda:
                        hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                    else:
                        hidden_states_for_token_i = layer[:, i, :].numpy()
                    all_layers_matrix_as_list[k].append(hidden_states_for_token_i)

                del one_batch, tokens_tensor, segments_tensor, model_outputs

            for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
                init_matrix = np.zeros((l_tokens, l_tokens))

                for i, hidden_states in enumerate(one_layer_matrix):
                    base_state = hidden_states[i]

                    for j, state in enumerate(hidden_states):
                        if self.args.metric == 'dist':
                            init_matrix[i][j] = np.linalg.norm(base_state - state)
                        if self.args.metric == 'cos':
                            init_matrix[i][j] = np.dot(base_state, state) / (
                                    np.linalg.norm(base_state) * np.linalg.norm(state))
                out[k].append((sentence, label, tokenized_text, eval(deprels), root_id, init_matrix))

        return out

    def extract_heads(self, matrix):

        results = {}

        for layer, state in tqdm(enumerate(matrix), total=len(matrix)):

            for s, (sentence, label, tokens, ud_deprels, root_id, sent_matrix) in enumerate(state):

                if s not in results:
                    results[s] = {'text': sentence,
                                  'label': label
                                  }
                if layer not in results[s]:
                    results[s][layer] = {}

                # map bpe tokens to words
                mapping = map_seq(tokens)
                init_matrix = sent_matrix

                # merge subwords in one row
                merge_column_matrix = []
                for i, line in enumerate(init_matrix):
                    new_row = []
                    buf = []
                    for j in range(0, len(line)):
                        buf.append(line[j])
                        if j != len(mapping) - 1:
                            if mapping[j] != mapping[j + 1]:
                                new_row.append(buf[0])
                        else:
                            new_row.append(buf[0])
                        buf = []
                    merge_column_matrix.append(new_row)

                # merge subwords in multi rows
                # transpose the matrix so we can work with row instead of multiple rows
                merge_column_matrix = np.array(merge_column_matrix).transpose()
                merge_column_matrix = merge_column_matrix.tolist()
                final_matrix = []
                for i, line in enumerate(merge_column_matrix):
                    new_row = []
                    buf = []
                    for j in range(0, len(line)):
                        buf.append(line[j])
                        if j != len(mapping) - 1:
                            if mapping[j] != mapping[j + 1]:
                                if self.args.subword == 'sum':
                                    new_row.append(sum(buf))
                                elif self.args.subword == 'avg':
                                    new_row.append((sum(buf) / len(buf)))
                                elif self.args.subword == 'first':
                                    new_row.append(buf[0])
                        else:
                            new_row.append(buf[0])
                        buf = []
                    final_matrix.append(new_row)

                # transpose to the original matrix
                final_matrix = np.array(final_matrix).transpose()

                # get root index and gold analysis
                # root_id, ud_heads = get_ud_analysis(sentence, self.ud_model)
                root = final_matrix[root_id].copy()

                # put root on 0 position
                final_matrix[root_id], final_matrix[0] = final_matrix[0], root

                perturbed_sent = merge_tokens(tokens)
                perturbed_sent[root_id], perturbed_sent[0] = perturbed_sent[0], perturbed_sent[root_id]

                # extract deprels
                heads, _ = chu_liu_edmonds(final_matrix)
                deprels = get_deprels(heads, perturbed_sent)

                if 'ud' not in results[s]:
                    results[s]['ud'] = {'deprels': ud_deprels}

                # to fix: when ud and bert tokenizations do not match
                if len(deprels) != len(ud_deprels):
                    continue

                # save results
                results[s][layer]['heads'] = heads
                results[s][layer]['deprels'] = deprels
                results[s][layer]['uas'] = uas(deprels, ud_deprels)
                results[s][layer]['uuas'] = uuas(deprels, ud_deprels)

        return results

    def get_attention_matrix(self):

        LAYER = int(self.args.layers)
        out = [[] for _ in range(LAYER)]

        for ind, (sentence, tokens, label, deprels, root_id) in tqdm(
                enumerate(self.data), total=len(self.data)):

            # Convert token to vocabulary indices
            tokenized_text = list(tokens)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            if self.args.cuda:
                tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            else:
                tokens_tensor = torch.tensor([indexed_tokens])

            # get hidden states
            with torch.no_grad():
                model_outputs = self.model(tokens_tensor)
                attentions = model_outputs.attentions  # 12 layers

            for k, layer in enumerate(attentions):
                if self.args.cuda:
                    attentions_for_layer = layer[0, :, :].cpu().numpy()
                else:
                    attentions_for_layer = layer[0, :, :].numpy()
                out[k].append((sentence, label, tokens, eval(deprels), root_id, attentions_for_layer))

        return out

    def extract_heads_attention(self, matrix):
        results = {}

        for layer, state in tqdm(enumerate(matrix), total=len(matrix)):

            for s, (sentence, label, tokens, ud_deprels, root_id, matrices) in enumerate(state):

                if s not in results:
                    results[s] = {}
                    results[s]['text'] = sentence
                    results[s]['label'] = label
                if layer not in results[s]:
                    results[s][layer] = {}

                # put root at position 0 for dependecy extraction
                perturbed_sent = merge_tokens(tokens)
                perturbed_sent[root_id], perturbed_sent[0] = perturbed_sent[0], perturbed_sent[root_id]

                if 'ud' not in results[s]:
                    results[s]['ud'] = {'deprels': ud_deprels}

                for head, sent_matrix in enumerate(matrices):

                    if head not in results[s][layer]:
                        results[s][layer][head] = {}

                    mapping = map_seq(tokens)
                    init_matrix = sent_matrix

                    # merge subwords in one row
                    merge_column_matrix = []
                    for i, line in enumerate(init_matrix):
                        new_row = []
                        buf = []
                        for j in range(0, len(line)):
                            buf.append(line[j])
                            if j != len(mapping) - 1:
                                if mapping[j] != mapping[j + 1]:
                                    new_row.append(buf[0])
                            else:
                                new_row.append(buf[0])
                            buf = []
                        merge_column_matrix.append(new_row)

                    # merge subwords in multi rows
                    # transpose the matrix so we can work with row instead of multiple rows
                    merge_column_matrix = np.array(merge_column_matrix).transpose()
                    merge_column_matrix = merge_column_matrix.tolist()
                    final_matrix = []
                    for i, line in enumerate(merge_column_matrix):
                        new_row = []
                        buf = []
                        for j in range(0, len(line)):
                            buf.append(line[j])
                            if j != len(mapping) - 1:
                                if mapping[j] != mapping[j + 1]:
                                    if self.args.subword == 'sum':
                                        new_row.append(sum(buf))
                                    elif self.args.subword == 'avg':
                                        new_row.append((sum(buf) / len(buf)))
                                    elif self.args.subword == 'first':
                                        new_row.append(buf[0])
                            else:
                                new_row.append(buf[0])
                            buf = []
                        final_matrix.append(new_row)

                    # transpose to the original matrix
                    final_matrix = np.array(final_matrix).transpose()

                    # ignore attention to self
                    final_matrix[range(final_matrix.shape[0]), range(final_matrix.shape[0])] = 0

                    # put root at position 0 for tree extraction
                    root = final_matrix[root_id].copy()
                    final_matrix[root_id], final_matrix[0] = final_matrix[0], root

                    final_matrix = np.array(final_matrix, dtype='double')

                    # extract deprels
                    heads, _ = chu_liu_edmonds(final_matrix)

                    deprels = get_deprels(heads, perturbed_sent)

                    # to fix: when ud and bert tokenizations do not match
                    if len(deprels) != len(ud_deprels):
                        continue

                    # save results
                    results[s][layer][head]['heads'] = heads
                    results[s][layer][head]['deprels'] = deprels
                    results[s][layer][head]['uas'] = uas(deprels, ud_deprels)
                    results[s][layer][head]['uuas'] = uuas(deprels, ud_deprels)

        return results

    def run_perturbed_probe(self):
        set_seed()
        print('Extracting dependency matrix...')
        matrix = self.get_dep_matrix()
        print('Extracting trees...')
        return self.extract_heads(matrix)

    def run_attentions_probe(self):
        set_seed()
        print('Extracting attention matrix...')
        matrix = self.get_attention_matrix()
        print('Extracting trees...')
        return self.extract_heads_attention(matrix)
