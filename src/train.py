import model

class TrainModel:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def train(self, num_epochs=10):
        # training loop
        pass
