import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
import re
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from underthesea import word_tokenize
import sys

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f"Thiết bị sử dụng: {device}")

def remove_emoji(text):
    for emoji in UNICODE_EMOJI.values():
        text = text.replace(emoji, "")
    for emoticon in EMOTICONS_EMO.values():
        text = text.replace(emoticon, "")
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', ' <num> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"[!@#$\[\]()']", "", text)

    # Load stopwords once and use set for faster lookup
    with open('vietnamese-stopwords.txt', "r", encoding="utf-8") as f:
        stopwords = set(f.read().split("\n"))
    
    words = word_tokenize(text)
    text = " ".join(word for word in words if word not in stopwords)
    return text

def preprocess_input(sentence, tokenizer, max_length=128):
    sentence = remove_emoji(sentence)
    sentence = clean_text(sentence)
    tokens = tokenizer(sentence, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return tokens['input_ids']


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, max_length):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the last output of the LSTM
        return out

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Mô hình đã được nạp từ {path}")

def test_model(model, tokenizer, sentence, device):
    model.eval()
    input_ids = preprocess_input(sentence, tokenizer).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted_label = torch.max(outputs, 1)
    label_map = {0: "Tích cực (POS)", 1: "Trung tính (NEU)", 2: "Tiêu cực (NEG)"}
    return label_map[predicted_label.item()]

# Đường dẫn đến mô hình và tokenizer
checkpoint = 'distilbert-base-multilingual-cased'
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)

vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_size = 64
output_size = 3
max_length = 128

model = LSTMModel(vocab_size, embedding_dim, hidden_size, output_size, max_length).to(device)

best_model_path = "./model/best_lstm_model.pth"
load_model(model, best_model_path, device)

# # Đọc file CSV
# input_csv_path = "post_facebook_ued_confessions.csv"  # Thay bằng đường dẫn đến file CSV của bạn
# output_csv_path = "output.csv"


def process_csv(input_csv_path, output_csv_path):
    # Đọc dữ liệu
    df = pd.read_csv(input_csv_path)
    
    # Kiểm tra và xử lý
    if 'content' in df.columns:
        predictions = []
        for content in df['content']:
            prediction = test_model(model, tokenizer, content, device)
            predictions.append(prediction)
        df['prediction'] = predictions
        df.to_csv(output_csv_path, index=False)
    else:
        print("Cột 'content' không tồn tại trong file CSV.")
        sys.exit(1)

if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    process_csv(input_csv, output_csv)