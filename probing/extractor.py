import warnings

warnings.simplefilter(action="ignore", category=(FutureWarning, DeprecationWarning))

import json
import numpy as np
import torch
from tqdm.auto import tqdm
from probing.utilities import *
from ufal.chu_liu_edmonds import chu_liu_edmonds
import pickle
from collections import defaultdict


class PerturbedProbe(object):
    def __init__(self, model, tokenizer, data, dataset, args):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.dataset = dataset
        self.args = args
        self.layers = self.args.layers + 1

        if self.args.cuda:
            self.model.to('cuda')
        self.model.eval()

    def __extract_one_matrix(self, root_id, init_matrix, mapping, perturbed_sent):

        # merge subwords in one row
        merge_column_matrix = []
        for i, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
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
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    if self.args.subword == 'sum':
                        new_row.append(sum(buf))
                    elif self.args.subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif self.args.subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            final_matrix.append(new_row)

        final_matrix = np.array(final_matrix).transpose()
        # transpose to the original matrix
        if self.args.model not in ["facebook/mbart-large-cc25"]:
             final_matrix = final_matrix[1:, 1:]

        # ignore attention to self
        final_matrix[range(final_matrix.shape[0]), range(final_matrix.shape[0])] = 0
        # put root at position 0 for tree extraction
        root = final_matrix[root_id].copy()
        final_matrix[root_id], final_matrix[0] = final_matrix[0], root
        root = final_matrix[:,root_id].copy()
        final_matrix[:,root_id], final_matrix[:,0] = final_matrix[:,0], root

        assert final_matrix.shape == (len(perturbed_sent), len(perturbed_sent))

        return final_matrix
    
    def __perturbe_sent(self, label, sentence, root_id, ud_deprels, out):

        mask_id = self.tokenizer.mask_token_id
        
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.encode(sentence)
        tokenized_text = self.tokenizer.convert_ids_to_tokens(indexed_tokens)

        ud_sent = sentence.split()
        
        if '[UNK]' in tokenized_text or '<unk>' in tokenized_text:
            return out

        # map tokens to words
        if self.args.model in ["bert-base-multilingual-cased"]:
            mapping = map_seq_bert(tokenized_text, ud_sent)
        else:
            mapping = map_seq_xlmr(tokenized_text, ud_sent)
        
        # put root at position 0 for dependecy extraction
        if self.args.model in ["facebook/mbart-large-cc25"]:
            perturbed_sent = merge_tokens(tokenized_text, mapping)[:-1]
        elif self.args.model in ["bert-base-multilingual-cased"]:
            perturbed_sent = merge_tokens(tokenized_text, mapping)[1:-1]
        elif self.args.model in ["xlm-roberta-base"]:
            perturbed_sent = merge_tokens(tokenized_text, mapping)[1:-1]
        
        assert perturbed_sent == ud_sent

        perturbed_sent[root_id], perturbed_sent[0] = perturbed_sent[0], perturbed_sent[root_id]
        
        # 1. Generate mask indices
        all_layers_matrix_as_list = [[] for i in range(self.layers)]
        for i in range(0, len(tokenized_text)):
            id_for_all_i_tokens = get_subwords(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            for tmp_id in id_for_all_i_tokens:
                if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                    tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
            for j in range(0, len(tokenized_text)):
                id_for_all_j_tokens = get_subwords(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                    if mapping[tmp_id] != -1:
                        one_batch[j][tmp_id] = mask_id
        
            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(one_batch)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
            if self.args.cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')

            # 3. get all hidden states for one batch    
            with torch.no_grad():
                if self.args.model in ["facebook/mbart-large-cc25"]:
                    model_outputs = self.model(tokens_tensor)
                    all_layers = model_outputs.encoder_hidden_states
                elif self.args.model in ['bert-base-multilingual-cased']:
                    model_outputs = self.model(tokens_tensor, segments_tensor)
                    all_layers = model_outputs.hidden_states
                elif self.args.model in ['xlm-roberta-base']:
                    model_outputs = self.model(tokens_tensor)
                    all_layers = model_outputs.hidden_states

            
            # 4. get hidden states for word_i in one batch
            for k, layer in enumerate(all_layers):
                if self.args.cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
                    hidden_states_for_token_i = layer[:, i, :].numpy()
                all_layers_matrix_as_list[k].append(hidden_states_for_token_i)
        

        for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
            init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
            for i, hidden_states in enumerate(one_layer_matrix):
                base_state = hidden_states[i]
                for j, state in enumerate(hidden_states):
                    if self.args.metric == 'dist':
                        init_matrix[i][j] = np.linalg.norm(base_state - state)
                    if self.args.metric == 'cos':
                        init_matrix[i][j] = np.dot(base_state, state) / (
                                    np.linalg.norm(base_state) * np.linalg.norm(state))
                        

            init_matrix = self.__extract_one_matrix(root_id, init_matrix, mapping, perturbed_sent)

            if label == 'O':
                out[k].append([(sentence, perturbed_sent, root_id, init_matrix, ud_deprels)])
            elif label == 'I':
                out[k][-1].append((sentence, perturbed_sent, root_id, init_matrix))

        return out


    def run(self):
        out = [[] for layer in range(self.layers)]

        for (label, correct_sent, incorrect_sent, cor_root, inc_root, ud_deprels) in tqdm(self.data, total=len(self.data)):

            out = self.__perturbe_sent('O', correct_sent, cor_root,  ud_deprels, out)
            out = self.__perturbe_sent('I', incorrect_sent, inc_root, ud_deprels, out)
        
        create_results_directory(save_dir_name='matrices',
                                 task=self.dataset,
                                 model=self.args.model,
                                 probe=self.args.prober)
        
        filename = 'results/matrices/{task}/{model}/{probe}/layer_{layer}.pkl'
        for k, one_layer_out in enumerate(out):

            k_output = filename.format(task=self.dataset,
                                       model=self.args.model,
                                       probe=self.args.prober,
                                       layer=str(k))
            
            with open(k_output, 'wb') as fout:
                pickle.dump(out[k], fout)
                fout.close()


    def evaluate(self):
        layer_results = [{'uuas': [],
                          'delta uuas': [],
                          'iou': [],
                          'delta iou': [],
                          'l2': []} for _ in range(self.layers)]
    
        filename = 'results/matrices/{task}/{model}/{probe}/layer_{layer}.pkl'
        for l in tqdm(range(self.layers), total=self.layers):

            k_output = filename.format(task=self.dataset,
                                       model=self.args.model,
                                       probe=self.args.prober,
                                       layer=str(l))
            
            with open(k_output, 'rb') as f:
                results = pickle.load(f)
            
            for sentence in results:

                if len(sentence) != 2:
                    continue

                cor_sent, cor_tokens, cor_root_id, cor_matrix, gold_tree = sentence[0]
                inc_sent, inc_tokens, inc_root_id, inc_matrix = sentence[1]

               # extract trees
                cor_heads, _ = chu_liu_edmonds(cor_matrix)
                cor_deprels = get_deprels(cor_heads, cor_tokens)
                assert len(cor_deprels) == len(gold_tree)

                inc_heads, _ = chu_liu_edmonds(inc_matrix)
                inc_deprels = get_deprels(inc_heads, inc_tokens)
                assert len(inc_deprels) == len(gold_tree)
                    
                # calculate uuas
                cor_uuas = uuas(cor_deprels, gold_tree)
                inc_uuas = uuas(inc_deprels, gold_tree)
                # calculate intersection over union
                cor_iou = iou(cor_deprels, gold_tree)
                inc_iou = iou(inc_deprels, gold_tree)

                # sort matrices in alphabetic order
                cor_matrix = sort_matrix(cor_matrix, cor_tokens)
                inc_matrix = sort_matrix(inc_matrix, inc_tokens)
                assert cor_matrix.shape == inc_matrix.shape

                # calculate l2 distance
                l2 = np.sum(np.power((cor_matrix-inc_matrix),2))

                # save results
                layer_results[l]['uuas'].append((cor_uuas, inc_uuas))
                layer_results[l]['delta uuas'].append(cor_uuas - inc_uuas)
                layer_results[l]['iou'].append((cor_iou, inc_iou))
                layer_results[l]['delta iou'].append(cor_iou - inc_iou)
                layer_results[l]['l2'].append(l2)

            # find means for layers   
            layer_results[l]['uuas'] = (
                np.mean([i[0] for i in layer_results[l]['uuas']]),
                np.mean([i[1] for i in layer_results[l]['uuas']])
            )
            layer_results[l]['iou'] = (
                np.mean([i[0] for i in layer_results[l]['iou']]),
                np.mean([i[1] for i in layer_results[l]['iou']])
            )

            layer_results[l]['delta uuas'] = np.mean(layer_results[l]['delta uuas'])
            layer_results[l]['delta iou'] = np.mean(layer_results[l]['delta iou'])
            layer_results[l]['l2'] = np.mean(layer_results[l]['l2'])

            # print('layer:', l)
            # print('delta uuas:', layer_results[l]['delta uuas'])
            # print('delta iou:', layer_results[l]['delta iou'])
            # print('l2:', layer_results[l]['l2'])

            #     print('layer:', l, 'head:', h)
            #     print('delta uuas:', head_results[l][h]['delta uuas'])
            #     print('delta iou:', head_results[l][h]['delta iou'])
            #     print('l2:', head_results[l][h]['l2'])

        return layer_results


class AttentionProbe(object):
    def __init__(self, model, tokenizer, data, dataset, args):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
        self.layers = self.args.layers

        if self.args.model in ["facebook/mbart-large-cc25"]:
            self.heads = 16
        else:
            self.heads = 12

        if self.args.cuda:
            self.model.to('cuda')

        self.model.eval()

    def __extract_one_matrix(self, root_id, attentions, mapping, perturbed_sent):

        final_attention = []

        for head, init_matrix in enumerate(attentions):
            
            # merge subwords in one row
            merge_column_matrix = []
            for i, line in enumerate(init_matrix):
                new_row = []
                buf = []
                for j in range(0, len(line) - 1):
                    buf.append(line[j])
                    if mapping[j] != mapping[j + 1]:
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
                for j in range(0, len(line) - 1):
                    buf.append(line[j])
                    if mapping[j] != mapping[j + 1]:
                        if self.args.subword == 'sum':
                            new_row.append(sum(buf))
                        elif self.args.subword == 'avg':
                            new_row.append((sum(buf) / len(buf)))
                        elif self.args.subword == 'first':
                            new_row.append(buf[0])
                        buf = []
                final_matrix.append(new_row)
            
            final_matrix = np.array(final_matrix).transpose()
            # transpose to the original matrix
            if self.args.model not in ["facebook/mbart-large-cc25"]:
                final_matrix = final_matrix[1:, 1:]

            # ignore attention to self
            final_matrix[range(final_matrix.shape[0]), range(final_matrix.shape[0])] = 0

            # gold root
            root = final_matrix[root_id].copy()
            final_matrix[root_id], final_matrix[0] = final_matrix[0], root
            root = final_matrix[:,root_id].copy()
            final_matrix[:,root_id], final_matrix[:,0] = final_matrix[:,0], root

            final_matrix = np.array(final_matrix, dtype='double')
            
            assert final_matrix.shape == (len(perturbed_sent), len(perturbed_sent))

            final_attention.append(final_matrix)

        return np.array(final_attention)


    def attend(self, label, sentence, root_id, ud_deprels, out):

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.encode(sentence)
        tokenized_text = self.tokenizer.convert_ids_to_tokens(indexed_tokens)
        segments_tensor = torch.tensor([[0 for _ in indexed_tokens]])

        # additional check for unknown tokens
        if '[UNK]' in tokenized_text or '<unk>' in tokenized_text:
            return out

        ud_sent = sentence.split()
        
        # map tokens to words
        if self.args.model in ["bert-base-multilingual-cased"]:
            mapping = map_seq_bert(tokenized_text, ud_sent)
        else:
            mapping = map_seq_xlmr(tokenized_text, ud_sent)

        # put root at position 0 for dependecy extraction
        if self.args.model in ["facebook/mbart-large-cc25"]:
            perturbed_sent = merge_tokens(tokenized_text, mapping)[:-1]
        elif self.args.model in ["bert-base-multilingual-cased"]:
            perturbed_sent = merge_tokens(tokenized_text, mapping)[1:-1]
        elif self.args.model in ["xlm-roberta-base"]:
            perturbed_sent = merge_tokens(tokenized_text, mapping)[1:-1]

        assert perturbed_sent == ud_sent

        perturbed_sent[root_id], perturbed_sent[0] = perturbed_sent[0], perturbed_sent[root_id]

        if self.args.cuda:
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            segments_tensor = segments_tensor.to('cuda')
        else:
            tokens_tensor = torch.tensor([indexed_tokens])
            
        # get attention matrices
        with torch.no_grad():
            if self.args.model in ["facebook/mbart-large-cc25"]:
                model_outputs = self.model(tokens_tensor)
                attentions = model_outputs.encoder_attentions
            elif self.args.model in ['bert-base-multilingual-cased']:
                model_outputs = self.model(tokens_tensor, segments_tensor)
                attentions = model_outputs.attentions
            elif self.args.model in ['xlm-roberta-base']:
                model_outputs = self.model(tokens_tensor)
                attentions = model_outputs.attentions

        for k, layer in enumerate(attentions):
            if self.args.cuda:
                attentions_for_layer = layer[0, :, :].cpu().numpy()
            else:
                attentions_for_layer = layer[0, :, :].numpy()

            attentions_for_layer = self.__extract_one_matrix(root_id, attentions_for_layer, mapping, perturbed_sent)

            if label == 'O':
                out[k].append([(sentence, perturbed_sent, root_id, attentions_for_layer, ud_deprels)])
            elif label == 'I':
                out[k][-1].append((sentence, perturbed_sent, root_id, attentions_for_layer))

        return out

    def run(self):

        out = [[] for layer in range(self.layers)]

        for (label, correct_sent, incorrect_sent, cor_root, inc_root, ud_deprels) in tqdm(
                self.data, total=len(self.data)):
            
            out = self.attend('O', correct_sent, cor_root,  ud_deprels, out)
            out = self.attend('I', incorrect_sent, inc_root, ud_deprels, out)

        create_results_directory(save_dir_name='matrices',
                                 task=self.dataset,
                                 model=self.args.model,
                                 probe=self.args.prober)
        
        filename = 'results/matrices/{task}/{model}/{probe}/layer_{layer}.pkl'
        for k, one_layer_out in enumerate(out):

            k_output = filename.format(task=self.dataset,
                                       model=self.args.model,
                                       probe=self.args.prober,
                                       layer=str(k))
            
            with open(k_output, 'wb') as fout:
                pickle.dump(out[k], fout)
                fout.close()

    def evaluate(self):

        layer_results = [{'uuas': [],
                          'delta uuas': [],
                          'iou': [],
                          'delta iou': [],
                          'l2': []} for _ in range(self.layers)]
        head_results = [[{'uuas': [],
                          'delta uuas': [],
                          'iou': [],
                          'delta iou': [],
                          'l2': []} for _ in range(self.heads)] for _ in range(self.layers)]

        filename = 'results/matrices/{task}/{model}/{probe}/layer_{layer}.pkl'
        for l in tqdm(range(self.layers), total=self.layers):

            k_output = filename.format(task=self.dataset,
                                       model=self.args.model,
                                       probe=self.args.prober,
                                       layer=str(l))
            
            with open(k_output, 'rb') as f:
                results = pickle.load(f)
            
            for sentence in results:
                if len(sentence) != 2:
                    continue

                cor_sent, cor_tokens, cor_root_id, cor_attentions, gold_tree = sentence[0]
                inc_sent, inc_tokens, inc_root_id, inc_attentions = sentence[1]

                for h in range(self.heads):
                    
                    # extract trees
                    cor_heads, _ = chu_liu_edmonds(cor_attentions[h])
                    cor_deprels = get_deprels(cor_heads, cor_tokens)
                    assert len(cor_deprels) == len(gold_tree)

                    inc_heads, _ = chu_liu_edmonds(inc_attentions[h])
                    inc_deprels = get_deprels(inc_heads, inc_tokens)
                    assert len(inc_deprels) == len(gold_tree)
                    
                    # calculate uuas
                    cor_uuas = uuas(cor_deprels, gold_tree)
                    inc_uuas = uuas(inc_deprels, gold_tree)
                    # calculate intersection over union
                    cor_iou = iou(cor_deprels, gold_tree)
                    inc_iou = iou(inc_deprels, gold_tree)

                    # sort matrices in alphabetic order
                    cor_matrix = sort_matrix(cor_attentions[h], cor_tokens)
                    inc_matrix = sort_matrix(inc_attentions[h], inc_tokens)
                    assert cor_matrix.shape == inc_matrix.shape

                    # calculate l2 distance
                    l2 = np.sum(np.power((cor_matrix-inc_matrix),2))

                    # save results
                    layer_results[l]['uuas'].append((cor_uuas, inc_uuas))
                    layer_results[l]['delta uuas'].append(cor_uuas - inc_uuas)
                    layer_results[l]['iou'].append((cor_iou, inc_iou))
                    layer_results[l]['delta iou'].append(cor_iou - inc_iou)
                    layer_results[l]['l2'].append(l2)

                    head_results[l][h]['uuas'].append((cor_uuas, inc_uuas))
                    head_results[l][h]['delta uuas'].append(cor_uuas - inc_uuas)
                    head_results[l][h]['iou'].append((cor_iou, inc_iou))
                    head_results[l][h]['delta iou'].append(cor_iou - inc_iou)
                    head_results[l][h]['l2'].append(l2)

            # find means for layers   
            layer_results[l]['uuas'] = (
                np.mean([i[0] for i in layer_results[l]['uuas']]),
                np.mean([i[1] for i in layer_results[l]['uuas']])
            )
            layer_results[l]['iou'] = (
                np.mean([i[0] for i in layer_results[l]['iou']]),
                np.mean([i[1] for i in layer_results[l]['iou']])
            )

            layer_results[l]['delta uuas'] = np.mean(layer_results[l]['delta uuas'])
            layer_results[l]['delta iou'] = np.mean(layer_results[l]['delta iou'])
            layer_results[l]['l2'] = np.mean(layer_results[l]['l2'])


            for h in range(12):
                head_results[l][h]['uuas'] = (
                    np.mean([i[0] for i in head_results[l][h]['uuas']]),
                    np.mean([i[1] for i in head_results[l][h]['uuas']])
                )
                head_results[l][h]['iou'] = (
                    np.mean([i[0] for i in head_results[l][h]['iou']]),
                    np.mean([i[1] for i in head_results[l][h]['iou']])
                )
                head_results[l][h]['delta uuas'] = np.mean(head_results[l][h]['delta uuas'])
                head_results[l][h]['delta iou'] = np.mean(head_results[l][h]['delta iou'])
                head_results[l][h]['l2'] = np.mean(head_results[l][h]['l2'])
                
            # print('layer:', l)
            # print('delta uuas:', layer_results[l]['delta uuas'])
            # print('delta iou:', layer_results[l]['delta iou'])
            # print('l2:', layer_results[l]['l2'])

            #     print('layer:', l, 'head:', h)
            #     print('delta uuas:', head_results[l][h]['delta uuas'])
            #     print('delta iou:', head_results[l][h]['delta iou'])
            #     print('l2:', head_results[l][h]['l2'])

        return layer_results, head_results

            
