import torch
import torch.nn as nn
from transformers import BertModel

class HybridModel(nn.Module):
    def __init__(self, num_numeric_features=6, num_classes=3):
        super(HybridModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)

        # For numeric features
        self.numeric_fc = nn.Sequential(
            nn.Linear(num_numeric_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.dropout(bert_output.pooler_output)
        numeric_feat = self.numeric_fc(numeric_features)

        combined = torch.cat((text_feat, numeric_feat), dim=1)
        return self.classifier(combined)
