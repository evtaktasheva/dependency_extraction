class Args:
    def __init__(self):
        self.prober = 'attention'
        self.cuda = True
        self.layers = 12
        self.metric = 'dist'
        self.subword = 'sum'
        self.model = 'bert-base-multilingual-cased'  #'DeepPavlov/rubert-base-cased' #'bert-base-cased'
        self.dataset = 'en_bigram_shift'  #'ngram_shift_O'  #'ngram_shift_deprels'
