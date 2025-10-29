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
    
def _process_batch(model, batch, device):
    """Processes a single batch and returns model outputs."""
    model_inputs = {
        'input_ids': batch['input_ids'].to(device, non_blocking=True),
        'attention_mask': batch['attention_mask'].to(device, non_blocking=True)
    }
    
    # Add optional inputs based on model type
    if 'confidence_score' in batch:
        model_inputs['confidence_score'] = batch['confidence_score'].to(device, non_blocking=True)
        
    if 'acoustic_feature' in batch:
        model_inputs['acoustic_feature'] = batch['acoustic_feature'].to(device, non_blocking=True)

    outputs = model(**model_inputs)
    return outputs

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device, gradient_clip_norm=0.0):
    """Performs a single training epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        outputs = _process_batch(model, batch, device)
        labels = batch['label'].to(device, non_blocking=True)
        
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        # Apply gradient clipping to prevent exploding gradients
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
        optimizer.step()
        scheduler.step()
        
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device, loss_fn=None):
    """Performs model evaluation on a given dataset."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            outputs = _process_batch(model, batch, device)
            labels = batch['label'].to(device, non_blocking=True)
            
            if loss_fn:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if loss_fn:
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy
    
    return all_preds, all_probs
