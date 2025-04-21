import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ========== CONFIG ==========
DATA_PATH = "nnindian_stock_news.csv"
MODEL_DIR = "models"
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUMERIC_COLS = ['PriceBefore', 'PriceAfter', 'ChangePercent', 'MA5', 'MA10', 'Volatility']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== GPU CHECK ==========
print("🔍 CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("💻 GPU in use:", torch.cuda.get_device_name(0))

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_PATH)
print(f"📊 Dataset loaded with {len(df)} rows")
df.dropna(subset=NUMERIC_COLS, inplace=True)

# Encode labels
le = LabelEncoder()
df['LabelEncoded'] = le.fit_transform(df['Label'])

# Normalize numeric features
scaler = StandardScaler()
df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(df['LabelEncoded']),
                                     y=df['LabelEncoded'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ========== DATASET ==========
class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe['Headline'].tolist()
        self.nums = dataframe[NUMERIC_COLS].values
        self.labels = dataframe['LabelEncoded'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding='max_length', truncation=True,
                             max_length=64, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numeric': torch.tensor(self.nums[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ========== STRATIFIED SPLIT ==========
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['LabelEncoded'], random_state=42)
train_dataset = NewsDataset(train_df)
val_dataset = NewsDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ========== MODEL ==========
class HybridModel(nn.Module):
    def __init__(self, num_numeric_features, num_classes):
        super(HybridModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.4)
        self.fc_numeric = nn.Sequential(
            nn.Linear(num_numeric_features, 32),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask, numeric):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.dropout(bert_output.pooler_output)
        num_feat = self.fc_numeric(numeric)
        combined = torch.cat((text_feat, num_feat), dim=1)
        return self.classifier(combined)

# ========== INIT ==========
model = HybridModel(num_numeric_features=len(NUMERIC_COLS), num_classes=3).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ========== TRAIN ==========
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"\n🚀 Starting Epoch {epoch + 1}/{EPOCHS}")

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        numeric = batch['numeric'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, numeric)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Finished Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# ========== EVALUATE ==========
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="🔍 Validating"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        numeric = batch['numeric'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids, attention_mask, numeric)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Decode labels
pred_labels = le.inverse_transform(all_preds)
true_labels = le.inverse_transform(all_labels)

# Report
print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels))

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels, labels=le.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title("🧠 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ========== SAVE ==========
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "hybrid_sentiment_model.pth"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("\n💾 Model and preprocessing artifacts saved to /models")
