import pandas as pd
from collections import Counter
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
import re
from underthesea import word_tokenize
import sys

input_csv = sys.argv[1]
csv_file = input_csv  
df = pd.read_csv(csv_file)
def remove_emoji(text):
    for emoji in UNICODE_EMOJI.values():
        text = text.replace(emoji, "")
    for emoticon in EMOTICONS_EMO.values():
        text = text.replace(emoticon, "")
    return text

def clean_text(text):
    text = text.lower()
    text = remove_emoji(text)  
    text = re.sub(r'\d+', ' ', text)  
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    
    with open('vietnamese-stopwords.txt', "r", encoding="utf-8") as f:
        stopwords = set(f.read().split("\n"))
    
    # Tách từ và giữ lại cụm từ có ý nghĩa
    words = word_tokenize(text, format="text") 
    cleaned_words = [word for word in words.split() if word not in stopwords]  
    return " ".join(cleaned_words)

# Kết hợp và làm sạch toàn bộ văn bản
df['cleaned_content'] = df['content'].fillna('') + " " + df['comment'].fillna('')
df['cleaned_content'] = df['cleaned_content'].apply(clean_text)

full_text = " ".join(df['cleaned_content'])

# Tách cụm từ và đếm tần suất
phrases = full_text.split()
phrase_counts = Counter(phrases)

# Lấy top 100 cụm từ xuất hiện nhiều nhất
top_phrases = phrase_counts.most_common(100)

# In danh sách top 100 cụm từ
print("Top 100 cụm từ xuất hiện nhiều nhất:")
print(f"{'Cụm từ':<30} Tần suất")
print("-" * 40)
for phrase, count in top_phrases:
    print(f"{phrase:<30} {count}")
