import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import model


class EvaluateModel:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):

        # for calculating accuracy
        correct_preds = 0
        total_preds = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.test_loader):

                # use GPU if available
                data, labels = data.to(self.device), labels.to(self.device)

                predictions = self.model(data)
                predicted_classes = predictions.argmax(dim=1)
                correct_preds += (predicted_classes == labels).sum().item()
                total_preds += labels.size(0)
                all_preds.extend(predicted_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # return accuracy and confusion matrix
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, cm
