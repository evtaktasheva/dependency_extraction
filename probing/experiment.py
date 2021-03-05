from probing.extractor import ExtractDependencies
from probing.models_utils import LoadModels
from probing.utilities import DataLoader
from probing.utilities import save_results


class Experiment:
    def __init__(self, args, max_len=-1):
        self.args = args
        self.max_len = max_len

    def run(self):
        print('Loading models...')
        model_loader = LoadModels(self.args.model, self.args.prober)
        model = model_loader.load_model()
        tokenizer = model_loader.load_tokenizer()

        print('Loading data...')
        data = DataLoader(
            self.args.dataset, tokenizer=tokenizer,
            max_len=self.max_len
        ).load_data()

        print('* * ' * 30)
        print(f'Running {self.args.prober} on {self.args.dataset}...')
        extractor = ExtractDependencies(data=data, model=model, tokenizer=tokenizer, args=self.args)

        if self.args.prober == 'attention':
            results = extractor.run_attentions_probe()
        elif self.args.prober == 'perturbed':
            results = extractor.run_perturbed_probe()

        save_results(self.args.prober, results)