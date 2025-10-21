import torch
import torch.nn as nn
from transformers import BertModel

class BERTWithConfidence(nn.Module):
    """
    Custom BERT-based model that incorporates an ASR confidence score.
    It concatenates the hidden states of the last few BERT layers with a
    processed confidence score before final classification.
    """
    def __init__(self, model_name, num_classes):
        super(BERTWithConfidence, self).__init__()
        # Load the BERT model, specifying that we want the hidden states
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # A small network to process the confidence score (as in Figure 1 of the paper)
        # Input: 1 (the score), Output: a small embedding (e.g., 1024)
        confidence_embedding_size = 1024
        self.confidence_processor = nn.Sequential(
            nn.Linear(1, confidence_embedding_size),
            nn.ReLU()
        )
        
        # The paper concatenates the last 3 hidden states.
        # We take the [CLS] token representation from each.
        # The size of the input to the classifier is (BERT hidden size * 3) + (confidence embedding size)
        classifier_input_size = self.bert_hidden_size * 3 + confidence_embedding_size

        self.classifier = nn.Sequential(
            #nn.Dropout(0.1),
            nn.Linear(classifier_input_size, num_classes)
        )

    def forward(self, input_ids, attention_mask, confidence_score):
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        '''
        hidden_states = outputs.hidden_states # A tuple of all hidden states

        # Extract the last 3 hidden states
        last_three_layers = hidden_states[-3:]
        
        # Get the representation of the [CLS] token (at index 0) from each of these layers
        cls_tokens = [layer[:, 0, :] for layer in last_three_layers]
        
        # Concatenate the [CLS] token representations
        bert_output = torch.cat(cls_tokens, dim=1)
        #'''
        
        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = torch.sum(attention_mask, dim=1)
        
        batch_size = last_hidden_state.size(0)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        
        last_token_indices = sequence_lengths - 1
        second_last_token_indices = sequence_lengths - 2
        third_last_token_indices = sequence_lengths - 3
        
        last_hs = last_hidden_state[batch_indices, last_token_indices, :]
        second_last_hs = last_hidden_state[batch_indices, second_last_token_indices, :]
        third_last_hs = last_hidden_state[batch_indices, third_last_token_indices, :]

        bert_output = torch.cat([third_last_hs, second_last_hs, last_hs], dim=1)
        #'''

        # Process the confidence score
        # Add a dimension for the Linear layer: (batch_size) -> (batch_size, 1)
        confidence_output = self.confidence_processor(confidence_score.unsqueeze(1))

        # Concatenate the BERT output and the confidence score output
        combined_output = torch.cat([bert_output, confidence_output], dim=1)

        # Final classification
        logits = self.classifier(combined_output)
        
        return logits