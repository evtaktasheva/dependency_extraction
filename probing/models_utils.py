from transformers import AutoModel, MBartModel, AutoModelForMaskedLM, AutoConfig
from transformers import BertTokenizer, MBartTokenizer, XLMRobertaTokenizer
import torch
import numpy as np


class LoadModels(object):
    def __init__(self, model, prober, position_embedding):
        self.model = model
        self.prober = prober
        self.position_embedding = position_embedding

    def load_model(self):
        config = AutoConfig.from_pretrained(
            self.model,
            output_attentions=(True if self.prober == 'attention' else False),
            output_hidden_states=(True if self.prober == 'perturbed' else False),
            position_embedding_type=('absolute' if self.position_embedding == 'random' else self.position_embedding)
        )
        if self.prober == 'logprob':
            model = AutoModelForMaskedLM.from_pretrained(self.model)
        else:
            model = AutoModel.from_config(config)
            
        if self.position_embedding == 'random':
            pe = model.embeddings.position_embeddings.weight.data.numpy()
            model.embeddings.position_embeddings.weight.data = torch.tensor(np.random.random(pe.shape))
            model.embeddings.position_embeddings.weight.requires_grad = False
        return model

    def load_tokenizer(self):
        if self.model in ["xlm-roberta-base"]:
            return XLMRobertaTokenizer.from_pretrained(self.model, strip_accents=False)
        elif self.model in ['bert-base-multilingual-cased']:
            return BertTokenizer.from_pretrained(self.model, strip_accents=False)
        elif self.model in ['facebook/mbart-large-cc25']:
            return MBartTokenizer.from_pretrained('facebook/mbart-large-cc25', strip_accents=False)
