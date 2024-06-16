import pandas as pd
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time
import json
from tqdm import tqdm
import numpy as np

## QUANTITATIVE ANALYSIS

def process_date_column(data):
    def is_valid_date(date_str):
        try:
            pd.to_datetime(date_str, format='%Y-%m-%d')
            return True
        except ValueError:
            return False
    valid_date_mask = data['date'].apply(is_valid_date)
    data = data[valid_date_mask]
    data = data.reset_index(drop=True)
    data['date'] = pd.to_datetime(data['date']).dt.date
    return data

def process_daily_post_counts(platform, topic, output_file):
    # Load data from the CSV file
    input_file = os.path.join("../../data/clean", f"{platform}_{topic}.csv")
    data = pd.read_csv(input_file)
    
    data = process_date_column(data)

    # Group by date and count posts
    daily_counts = data.groupby('date').size().reset_index(name='posts')

    # Calculate cumulative posts
    daily_counts['cumulative_posts'] = daily_counts['posts'].cumsum()
    
    daily_counts['platform'] = platform
    daily_counts['topic'] = topic
    
    # Save the results to a CSV file
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        daily_counts = pd.concat([existing_data, daily_counts])

    daily_counts.to_csv(output_file, index=False)
        
def process_interaction_dist(platform, topic, output_file):
    # Load data from the CSV file
    input_file = os.path.join("../../data/clean", f"{platform}_{topic}.csv")
    data = pd.read_csv(input_file)
    
    # Group by interaction and count occurrences
    data = data.dropna(subset=['interaction'])
    data['interaction'] = data['interaction'].str.replace(',', '').astype(int)
    interaction_counts = data.groupby('interaction').size().reset_index(name='post_count')

    interaction_counts['platform'] = platform
    interaction_counts['topic'] = topic

    # Save the results to a CSV file
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        interaction_counts = pd.concat([existing_data, interaction_counts])

    interaction_counts.to_csv(output_file, index=False)
        
def process_daily_user_counts(platform, topic, output_file):
    # Load data from the CSV file
    input_file = os.path.join("../../data/clean", f"{platform}_{topic}.csv")
    data = pd.read_csv(input_file)

    data = process_date_column(data)

    # Filter out rows where author_id is null (if necessary)
    if 'author_id' in data.columns:
        data = data[data['author_id'].notnull()]

    # Count unique users up to each date
    data.sort_values(by=["date"], inplace=True)
    seen_users = set()
    results = []
    for date, group in data.groupby("date"):
        seen_users.update(group["author_id"].tolist())
        results.append({"date": date, "cumulative_unique_users": len(seen_users), "platform": platform, "topic": topic,})
    results = pd.DataFrame(results)

    # Save the results to a CSV file
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        results = pd.concat([existing_data, results])
    results.to_csv(output_file, index=False)
        
        
## SENTIMENT ANALYSIS

def process_sentiment(platform, topic,  output_file):
    sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest", device="cuda")
    
    data = pd.read_csv(f"../../data/clean/{platform}_{topic}.csv")
    data['clean_text'] = data['clean_text'].apply(lambda x: x if isinstance(x, str) else np.nan)
    data = data.dropna(subset=['clean_text'])
    #data = data.sample(n=10000, random_state=99)
    
    labels = []
    for instance in tqdm(data.itertuples(index=True, name='Pandas'), total=len(data)):
        text = str(instance.clean_text)[:300]
        labels.append(sentiment_task(text)[0]["label"])
        
    data["sentiment"] = labels
    data = data[["id", "clean_text", "sentiment"]]
    data.rename(columns={'clean_text': 'text'}, inplace=True)
    data['platform'] = platform
    data['topic'] = topic
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        data = pd.concat([existing_data, data])
    
    data.to_csv(output_file, index=False)
    
def process_emotion(platform, topic,  output_file):
    emotion_task = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", tokenizer="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", device="cuda")
    
    data = pd.read_csv(f"../../data/clean/{platform}_{topic}.csv")
    data['clean_text'] = data['clean_text'].apply(lambda x: x if isinstance(x, str) else np.nan)
    data = data.dropna(subset=['clean_text'])
    #data = data.sample(n=10000, random_state=99)
    
    labels = []
    for instance in tqdm(data.itertuples(index=True, name='Pandas'), total=len(data)):
        text = str(instance.clean_text)[:300]
        labels.append(emotion_task(text)[0]["label"])
        
    data["emotion"] = labels
    data = data[["id", "clean_text", "emotion"]]
    data.rename(columns={'clean_text': 'text'}, inplace=True)
    data['platform'] = platform
    data['topic'] = topic
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        data = pd.concat([existing_data, data])
    
    data.to_csv(output_file, index=False)

def process_LLM_conversation_type(platform, topic,  sentiment_data, emotion_data):
    LLM_output_path = f"../../data/LLM_output/{platform}/{topic}/output.json"
    
    labels = []
    with open(LLM_output_path, "r") as json_file:
        LLM_output_data = json.load(json_file)
        for instance in tqdm(LLM_output_data):
            if "IMAGE GENERATION" in instance["answer"].upper() and "EDUCATION" not in instance["answer"].upper() and "CREATIVE WRITING" not in instance["answer"].upper():
                sentiment_data.loc[sentiment_data['id'] == instance["instance_id"], 'category'] = "IMAGE_GENERATION"
                emotion_data.loc[emotion_data['id'] == instance["instance_id"], 'category'] = "IMAGE_GENERATION"
            elif "EDUCATION" in instance["answer"].upper() and "IMAGE GENERATION" not in instance["answer"].upper() and "CREATIVE WRITING" not in instance["answer"].upper():
                sentiment_data.loc[sentiment_data['id'] == instance["instance_id"], 'category'] = "EDUCATION"
                emotion_data.loc[emotion_data['id'] == instance["instance_id"], 'category'] = "EDUCATION"
            elif "CREATIVE WRITING" in instance["answer"].upper() and "IMAGE GENERATION" not in instance["answer"].upper() and "EDUCATION" not in instance["answer"].upper():
                sentiment_data.loc[sentiment_data['id'] == instance["instance_id"], 'category'] = "CREATIVE_WRITING"
                emotion_data.loc[emotion_data['id'] == instance["instance_id"], 'category'] = "CREATIVE_WRITING"
            else: continue
    
    return sentiment_data, emotion_data
    
def LLM_generation(platform, topic):
    
    PROMPT = """Question: The text "{text}" is a conversation about ChatGPT and Large Language Models. \nGiven the following conversation categories and their definitions: \n1) IMAGE GENERATION: comments about the use of advanced tools for creating visuals like DALLE-2, Midjourney or Stable Diffusion. \n2)EDUCATION: discussions about issues like plagiarism and AI-generated essays, students leveraging LLMs to cheat on assignments or LLMs utility in answering math and physics questions. \n3)CREATIVE WRITING: discussions related to various forms of written art, poetry, songs, screenplays and writing books.\n Outputs only one category for the above conversation. If you do not know the answer, do not answer. Do not motivate your anser. Answer: """
    
    output_file_path = f"../../data/LLM_output/{platform}/{topic}/"
    # to manage creation/deletion of folders
    if not os.path.exists(f"../../data/LLM_output/"):
        os.system(f"mkdir ../../data/LLM_output/")
    if not os.path.exists(f"../../data/LLM_output/{platform}/"):
        os.system(f"mkdir ../../data/LLM_output/{platform}/")
    if not os.path.exists(output_file_path):
        os.system(f"mkdir {output_file_path}")
    elif os.path.exists(f"{output_file_path}/output.txt"):
        t = 5
        while t > 0:
            _, secs = divmod(t, 60)
            timer = '{:02d}'.format(secs)
            print(f"\033[1mWarning\033[0m: Found output files in the target directory! I will delete them in {timer}", end='\r')
            time.sleep(1)
            t -= 1
        os.system(f"rm -r {output_file_path}/*")
        
    # let's generate outputs    
    data = pd.read_csv(f"../../data/clean/{platform}_{topic}.csv")
    # for timing reasons we only take 10000 samples randomly
    sampled_data = data.sample(n=10000, random_state=99)
    json_data = []
    
    full_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(full_model_name, load_in_4bit=True, device_map='auto')
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)

    
    with open(f"{output_file_path}/output.txt", "a") as fa_txt, open(f"{output_file_path}/output.json", "w") as fw_json:
        for instance in tqdm(sampled_data.itertuples(index=True, name='Pandas'), total=len(sampled_data)):
            
            instance_id = instance.id
            prompt = PROMPT.format(text=instance.text)

            answer = pipe(prompt)[0]["generated_text"].replace(prompt, "").replace("\n", "").strip()

            fa_txt.write(f"{instance_id}\t{answer}\n")
            fa_txt.flush()

            json_answer = {"instance_id":instance_id, "answer":answer}
            json_data.append(json_answer)
        json.dump(json_data, fw_json, indent=4)