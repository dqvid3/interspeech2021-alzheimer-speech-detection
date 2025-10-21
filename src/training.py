from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import numpy as np

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = -np.inf if mode == 'max' else np.inf

    def __call__(self, current_score):
        if self.mode == 'max':
            improvement = (current_score - self.best_score) > self.min_delta
        else:
            improvement = (self.best_score - current_score) > self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Early stop
        return False

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    """Performs a single training epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        confidence_score = batch['confidence_score'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            confidence_score=confidence_score
        )
        
        # Calculate the loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        # Apply gradient clipping to prevent exploding gradients
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update the model weights
        optimizer.step()
        # Update the learning rate scheduler
        scheduler.step()
        
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    """Performs model evaluation on a given dataset."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            confidence_score = batch['confidence_score'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                confidence_score=confidence_score
            )
            
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions (the class with the highest probability)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy