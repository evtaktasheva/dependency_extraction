class Args:
    def __init__(self):
        self.prober = 'attention'
        self.cuda = True
        self.layers = 12
        self.metric = 'dist'
        self.subword = 'sum'
        self.model = 'bert-base-multilingual-cased'
        self.probe_tasks = []
        self.position_embedding = 'absolute'
        self.distance = False
        self.token_embedding = 'default'
