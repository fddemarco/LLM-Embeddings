import zipfile



class ExperimentSuite:
    def __init__(self, model, strategy, experiments=None):
        self.model = model
        self.strategy = strategy
        self.experiments = experiments

    def run(self):
        for exp in self.experiments:
            data = exp.load_data()
            embeddings = None
            exp.save_embeddings(embeddings)
        self.compress(str(self.model))
        return self

    def compress(self, filename):
        with zipfile.ZipFile(f"{filename}.zip", mode="w") as archive:
            for exp in self.experiments:
                archive.write(exp.embeddings_filepath())
        return self
