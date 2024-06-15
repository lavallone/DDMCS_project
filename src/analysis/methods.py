import pandas as pd
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time
import argparse
import json
from tqdm import tqdm

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




if __name__ == "__main__":
    
    PROMPT = """Question: The text "{text}" is a conversation about ChatGPT and Large Language Models. \nGiven the following conversation categories and their definitions: \n1) IMAGE GENERATION: comments about the use of advanced tools for creating visuals like DALLE-2, Midjourney or Stable Diffusion. \n2)EDUCATION: discussions about issues like plagiarism and AI-generated essays, students leveraging LLMs to cheat on assignments or LLMs utility in answering math and physics questions. \n3)CREATIVE WRITING: discussions related to various forms of written art, poetry, songs, screenplays and writing books.\n Outputs only one category for the above conversation. If you do not know the answer, do not answer. Do not motivate your anser. Answer: """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", "-p", type=str)
    parser.add_argument("--topic", "-t", type=str)
    args = parser.parse_args()
    
    output_file_path = f"../../data/LLM_output/{args.platform}/{args.topic}/"
    # to manage creation/deletion of folders
    if not os.path.exists(f"../../data/LLM_output/"):
        os.system(f"mkdir ../../data/LLM_output/")
    if not os.path.exists(f"../../data/LLM_output/{args.platform}/"):
        os.system(f"mkdir ../../data/LLM_output/{args.platform}/")
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
    data = pd.read_csv(f"../../data/clean/{args.platform}_{args.topic}.csv")
    n_instances_processed = 0
    json_data = []
    
    full_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    #tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token
    #pipe = pipeline("text-generation", model=full_model_name, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)

    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(full_model_name, load_in_8bit=True, device_map='auto')
    pipe = pipeline("text-generation", model=model, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)

    
    with open(f"{output_file_path}/output.txt", "a") as fa_txt, open(f"{output_file_path}/output.json", "w") as fw_json:
        for instance in tqdm(data.itertuples(index=True, name='Pandas'), total=len(data)):
            
            instance_id = instance.id
            prompt = PROMPT.format(text=instance.text)

            answer = pipe(prompt)[0]["generated_text"].replace(prompt, "").replace("\n", "").strip()

            fa_txt.write(f"{instance_id}\t{answer}\n")
            fa_txt.flush()

            json_answer = {"instance_id":instance_id, "answer":answer}
            json_data.append(json_answer)
        json.dump(json_data, fw_json, indent=4)