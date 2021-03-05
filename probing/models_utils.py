from transformers import AutoModel, BertTokenizer, AutoConfig


class LoadModels(object):
    def __init__(self, model, prober):
        self.model = model
        self.prober = prober

    def load_model(self):
        config = AutoConfig.from_pretrained(
            self.model,
            output_attentions=(True if self.prober == 'attention' else False),
            output_hidden_states=(True if self.prober == 'perturbed' else False)
        )
        model = AutoModel.from_config(config)
        return model

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained(self.model, strip_accents=True)
