from transformers import AutoModel, MBartModel, AutoModelForMaskedLM, AutoConfig
from transformers import BertTokenizer, MBartTokenizer, XLMRobertaTokenizer
import torch
import numpy as np


class LoadModels(object):
    def __init__(self, model, prober, position_embedding, token_embedding):
        self.model = model
        self.prober = prober
        self.position_embedding = position_embedding
        self.token_embedding = token_embedding

    def load_model(self):
        config = AutoConfig.from_pretrained(
            self.model,
            output_attentions=(True if self.prober == 'attention' else False),
            output_hidden_states=(True if self.prober == 'perturbed' else False)
        )
        if self.prober == 'logprob':
            model = AutoModelForMaskedLM.from_pretrained(self.model)
        else:
            model = AutoModel.from_config(config)
            
        if self.position_embedding == 'random':
            model.embeddings.position_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        elif self.position_embedding == 'zero':
            model.embeddings.position_embeddings.weight.data.zero_()

        if self.token_embedding == 'random':
           model.embeddings.word_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        elif self.token_embedding == 'zero':
           model.embeddings.word_embeddings.weight.data.zero_()
        return model

    def load_tokenizer(self):
        if self.model in ["xlm-roberta-base"]:
            return XLMRobertaTokenizer.from_pretrained(self.model, strip_accents=False)
        elif self.model in ['bert-base-multilingual-cased']:
            return BertTokenizer.from_pretrained(self.model, strip_accents=False)
        elif self.model in ['facebook/mbart-large-cc25']:
            return MBartTokenizer.from_pretrained('facebook/mbart-large-cc25', strip_accents=False)
