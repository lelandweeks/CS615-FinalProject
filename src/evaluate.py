import torch
import model

class EvaluateModel:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

    def evaluate(self):

        # for calculating accuracy
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.test_loader):
                predictions = self.model(data)
                predicted_classes = predictions.argmax(dim=1)
                correct_preds += (predicted_classes == labels).sum().item()
                total_preds += labels.size(0)

        # return accuracy
        return correct_preds / total_preds if total_preds > 0 else 0
