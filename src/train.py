import time

import torch
import torch.nn as nn

class TrainModel:
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device

    def train(self, num_epochs=50, learning_rate=0.01):

        train_losses = []

        # loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # training loop
        for epoch in range(num_epochs):
            start = time.time()

            # needed to calculate average loss per epoch for training loss
            total_loss_per_epoch = 0
            
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                
                # use GPU if available
                data, labels = data.to(self.device), labels.to(self.device)

                # forward pass
                optimizer.zero_grad()
                predictions = self.model(data)
                loss = loss_fn(predictions, labels)

                # backward pass and optimization step
                loss.backward()
                optimizer.step()

                total_loss_per_epoch += loss.item()

            avg_loss_per_epoch = total_loss_per_epoch / len(self.train_loader)
            train_losses.append(avg_loss_per_epoch)

            elapsed = time.time() - start
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss_per_epoch:.4f}, Time: {elapsed:.2f}s")  

        return train_losses
