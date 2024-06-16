import pandas as pd
import re
from datetime import datetime
from transformers import pipeline
import torch
from tqdm import tqdm
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


def convert_date(date_str: str) -> datetime.date:
    # cleaning date string: removing timezone, we are interested in the date component.
    cleaned_str = re.sub(r"[A-Z]|T|Z", " ", date_str).strip()
    try:
        dt = pd.to_datetime(cleaned_str, errors="coerce")
        return dt.date()
    except:
        return None

def merge_text(data: pd.DataFrame) -> pd.DataFrame:
    text_columns = [col for col in data.columns if "text" in col.lower()]
    def process_text_cols(text_cols: pd.Series) -> str:
        text_values = [value for value in text_cols if pd.notna(value)]
        return " ".join(map(str, text_values))
    if len(text_columns) > 1:
        data["text"] = data[text_columns].apply(process_text_cols, axis=1)
        data = data.drop(columns=text_columns)
    else:
        data["text"] = data["text"].astype(str)
    return data

def remove_facebook_spam(data):
        """Detect and remove spam Facebook posts from the data."""
        
        spam_patterns = [
            "Video Funny Amazing #fyp #viral",
            "#reeel #cr7# #chatgpt",
            "#reels #chatgpt",
            "https://www.facebook.com/100076267686928/posts/202421482310107",
        ]
        data = data.copy()
        data["spam"] = 0
        data["text"] = data["text"].astype(str)
        # if a row's text contains any of the spam patterns, set spam = 1
        for pattern in spam_patterns:
            data.loc[data["text"].str.contains(pattern, case=False, na=False), "spam"] = 1
        # filter out spam posts
        data = data[data["spam"] == 0]
        data.drop(columns=["spam"], inplace=True)
        return data

def filter_language(data):
    language_model_detection = pipeline("text-classification",
                                        model="papluca/xlm-roberta-base-language-detection",
                                        tokenizer="papluca/xlm-roberta-base-language-detection",
                                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                        padding=True,
                                        truncation=True,
                                    )
    
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]
            
    def detect_language(texts):
        results = language_model_detection(texts)
        langs = [result["label"] for result in results]
        return langs
        
    texts = data["text"].tolist()
    langs = []
    for i, batch_texts in tqdm(enumerate(batch(texts, 64)), total=len(texts)//64):
        langs.extend(detect_language(batch_texts))
        if i % 25 == 0 and i > 0:
            print(f"processed {i * 64} rows - total {len(texts)}")
    
    data['language'] = langs
    data = data[data['language'] == 'en']
    data = data.drop(columns=['language'])
    return data
    
def clean_text(text):
    def base_cleaning(text):
        # remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+"
        )
        text = emoji_pattern.sub(r"", text)
        # remove URLs, HTML tags, and extra whitespaces
        text = re.sub(r"http\S+|www\S+|https\S+|<.*?>|\s+", " ", text, flags=re.MULTILINE)
        # remove string if it only contains punctuation
        #if text.strip(string.punctuation).strip() == "": return None
        # remove \r and \n
        text = re.sub(r"\r|\n", "", text)
        # remove 'x200b' and 'x200B' occurrences
        text = text.replace("x200b", "").replace("x200B", "")
        return text
    
    def lemmatize_text(text):
        
        def get_wordnet_pos(tag):
            if tag.startswith("J"): return wordnet.ADJ
            elif tag.startswith("V"): return wordnet.VERB
            elif tag.startswith("R"): return wordnet.ADV
            else: return wordnet.NOUN
            
        lemmatizer = WordNetLemmatizer()
        # tokenize and POS tag the text
        text = nltk.word_tokenize(text)
        pos_tagged_text = nltk.pos_tag(text)
        # lemmatize the text using POS tags
        text = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged_text]
        return text
    
    text = base_cleaning(text)
    if text is None: return None

    # remove punctuation
    #text = "".join(ch for ch in text if ch not in string.punctuation)

    text = lemmatize_text(text)

    #stop_words = set(stopwords.words("english"))
    #text = [word for word in text if word.lower() not in stop_words and not word.isdigit()]

    # last steps
    if len(text) < 2:
        return None
    text = text[:1000]
    text = " ".join(text).lower()
    return text