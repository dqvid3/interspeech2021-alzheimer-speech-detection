import torch
import torch.nn as nn
from transformers import BertModel

class ADBERTClassifier(nn.Module):
    """
    A unified BERT-based model for Alzheimer's detection that can operate in one of three modes:
    1. 'text_only': Standard BERT for sequence classification.
    2. 'confidence': Incorporates an ASR confidence score with text features.
    3. 'fusion': Fuses BERT linguistic features with external acoustic features.

    The mode is determined by the 'model_type' in the configuration.
    """
    def __init__(self, config):
        super(ADBERTClassifier, self).__init__()

        model_name = config['model_name']
        num_classes = config['num_classes']
        self.model_type = config['model_type']

        # Load BERT model
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        if self.model_type == 'fusion':
            fusion_config = config['fusion_model']
            acoustic_dim = fusion_config['acoustic_dim']
            reduced_acoustic_dim = fusion_config['reduced_acoustic_dim']
            
            self.acoustic_processor = nn.Sequential(
                nn.Linear(acoustic_dim, reduced_acoustic_dim),
                nn.ReLU(),
            )
            classifier_input_size = self.bert_hidden_size + reduced_acoustic_dim
            classifier_hidden_size = fusion_config['classifier_hidden_size']
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, classifier_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(classifier_hidden_size, num_classes)
            )

        elif self.model_type == 'confidence':
            confidence_embedding_size = 1024
            self.confidence_processor = nn.Sequential(
                nn.Linear(1, confidence_embedding_size),
                nn.ReLU(),
            )
            # Concatenation of last 3 [CLS] tokens + confidence embedding
            classifier_input_size = self.bert_hidden_size * 3 + confidence_embedding_size
            self.classifier = nn.Linear(classifier_input_size, num_classes)
        
        elif self.model_type == 'text_only':
            classifier_input_size = self.bert_hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, num_classes)
            )
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'.")

    def forward(self, input_ids, attention_mask, **kwargs):
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.hidden_states

        if self.model_type == 'fusion':
            linguistic_output = hidden_states[-1][:, 0, :]
            
            acoustic_feature = kwargs.get('acoustic_feature')
            acoustic_output = self.acoustic_processor(acoustic_feature)

            combined_output = torch.cat([linguistic_output, acoustic_output], dim=1)
            logits = self.classifier(combined_output)

        elif self.model_type == 'confidence':
            cls_outputs = [layer[:, 0, :] for layer in hidden_states[-3:]]
            bert_output = torch.cat(cls_outputs, dim=1)

            confidence_score = kwargs.get('confidence_score')
            # Add a dimension for the Linear layer: (batch_size) -> (batch_size, 1)
            confidence_output = self.confidence_processor(confidence_score.unsqueeze(1))
            
            combined_output = torch.cat([bert_output, confidence_output], dim=1)
            logits = self.classifier(combined_output)

        elif self.model_type == 'text_only':
            linguistic_output = hidden_states[-1][:, 0, :]
            logits = self.classifier(linguistic_output)
            
        return logits
