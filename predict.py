# import torch
# import numpy as np

# def predict(headline, numeric_features, model, tokenizer, scaler, label_encoder, device):
#     model.eval()
    
#     # Tokenize headline
#     encoding = tokenizer(headline, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)

#     # Ensure numeric features are tensor
#     if isinstance(numeric_features, np.ndarray):
#         numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).unsqueeze(0).to(device)
#     else:
#         numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, numeric_features=numeric_tensor)
#         _, pred_idx = torch.max(outputs, dim=1)

#     return label_encoder.inverse_transform([pred_idx.item()])[0]


# import torch
# import numpy as np

# def predict(headline, numeric_features, model, tokenizer, scaler, label_encoder, device):
#     model.eval()

#     # Tokenize the headline
#     encoding = tokenizer(
#         headline,
#         padding='max_length',
#         truncation=True,
#         max_length=64,
#         return_tensors="pt"
#     )
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)

#     # Prepare numeric features
#     if isinstance(numeric_features, np.ndarray):
#         numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).unsqueeze(0).to(device)
#     else:
#         numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).unsqueeze(0).to(device)

#     # Get model output
#     with torch.no_grad():
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             numeric_features=numeric_tensor
#         )
#         probs = torch.nn.functional.softmax(outputs, dim=1)
#         _, pred_idx = torch.max(probs, dim=1)

#     # Decode label and extract confidence
#     label = label_encoder.inverse_transform([pred_idx.item()])[0]
#     confidence = probs[0][pred_idx].item()

#     return label, round(confidence * 100, 2)






import torch
import numpy as np

def predict(headline, numeric_features, model, tokenizer, scaler, label_encoder, device):
    model.eval()

    # Tokenize the headline
    encoding = tokenizer(
        headline,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Prepare numeric features
    if isinstance(numeric_features, np.ndarray):
        numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).unsqueeze(0).to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_features=numeric_tensor
        )
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, pred_idx = torch.max(probs, dim=1)

    # Decode label and extract confidence
    label = label_encoder.inverse_transform([pred_idx.item()])[0]
    confidence = probs[0][pred_idx].item()

    return label, round(confidence * 100, 2)




