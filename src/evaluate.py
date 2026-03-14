import model

class EvaluateModel:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

    def evaluate(self):
        # evaluation logic
        pass
