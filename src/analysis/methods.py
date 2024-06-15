import pandas as pd
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import time
import argparse
import json

## QUANTITATIVE ANALYSIS

def process_daily_post_counts(platform, topic, output_file):
        # Load data from the CSV file
        input_file = os.path.join("../../data/clean", f"{platform}_{topic}.csv")
        data = pd.read_csv(input_file)

        # Ensure the date column is of datetime type
        data['date'] = pd.to_datetime(data['date']).dt.date

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

        # Convert the date column to datetime
        data['date'] = pd.to_datetime(data['date']).dt.date

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

PROMPTS = {
	"sentiment" : """Question: What is the sentiment of this text? \nText: {text} \nOptions: [ "strongly negative", "negative", "negative or neutral", "positive", "strongly positive"] \nAnswer: {answer}""",
	"emotion" : """Question: Which emotions from the options below are expressed in the following text? \nText: {text} \nOptions: [ "anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust" ] \nAnswer: {answer}""",
	"topic" : """Question: The text "{text}" is a conversation about ChatGPT and Large Language Models. \nGiven the following conversation categories and their definitions: \n1) IMAGE GENERATION: comments about the use of advanced tools for creating visuals like DALLE-2, Midjourney or Stable Diffusion. \n2)EDUCATION: discussions about issues like plagiarism and AI-generated essays, students leveraging LLMs to cheat on assignments or LLMs utility in answering math and physics questions. \n3)CREATIVE WRITING: discussions related to various forms of written art, poetry, songs, screenplays and writing books.\n Outputs only one category for the above conversation. If you do not know the answer, do not answer. Answer: """,
}


# def _prepare_finetuned_model(shortcut_model_name:str, checkpoint_path:str):
#     # load the original model first
#     full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
#     tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
#     base_model = AutoModelForCausalLM.from_pretrained(
#         full_model_name,
#         quantization_config=None,
#         device_map=None,
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16,
#     ).cuda()

#     # merge fine-tuned weights with the base model
#     peft_model_id = checkpoint_path
#     model = PeftModel.from_pretrained(base_model, peft_model_id)
#     model.merge_and_unload()
    
#    return tokenizer, model
    
def countdown(t):
    while t > 0:
        _, secs = divmod(t, 60)
        timer = '{:02d}'.format(secs)
        print(f"\033[1mWarning\033[0m: Found output files in the target directory! I will delete them in {timer}", end='\r')
        time.sleep(1)
        t -= 1

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", "-p", type=str)
    parser.add_argument("--topic", "-t", type=str)
    parser.add_argument("--type_", "-ty", type=str)
    args = parser.parse_args()
    
    output_file_path = f"../../data/LLM_output/{args.type_}/"#{platform}_{topic}.json"
    # to manage creation/deletion of folders
    if not os.path.exists(f"../../data/LLM_output/"):
        os.system(f"mkdir ../../data/LLM_output/")
    if not os.path.exists(output_file_path):
        os.system(f"mkdir {output_file_path}")
    elif os.path.exists(f"{output_file_path}/{args.platform}_{args.topic}_output.txt"):
        countdown(5)
        os.system(f"rm -r {output_file_path}/*")
        
    # let's generate outputs    
    data = pd.read_csv(f"../../data/clean/{args.platform}_{args.topic}.csv")
    n_instances_processed = 0
    json_data = []

    # if the model is finetuned, the checkpoint path is needed
    #CAPIRE COME QUANTIZZARE
    # if args.type_ == "emotion" or args.type_ == "sentiment":
    #     finetuned_tokenizer, finetuned_model = _prepare_finetuned_model(shortcut_model_name, checkpoint_path)
    #     finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
    #     pipe = pipeline("text-generation", model=finetuned_model, device="cuda", tokenizer=finetuned_tokenizer, pad_token_id=finetuned_tokenizer.eos_token_id, max_new_tokens=25)
    # else:
    full_model_name = "h2oai-h2o-danube2-1.8b-chat"
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline("text-generation", model=full_model_name, device="cuda", tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)

    # data è un panda frame (capire se funziona o meno)
    with open(f"{output_file_path}/{args.platform}_{args.topic}_output.txt", "a") as fa_txt, open(f"{output_file_path}/{args.platform}_{args.topic}_output.json", "w") as fw_json:
        for instance in data.iterrows():

            instance_id = instance["id"]
            prompt = PROMPTS[args.type_].format(text=instance["text"])

            answer = pipe(prompt)[0]["generated_text"].replace(prompt, "").replace("\n", "").strip()

            fa_txt.write(f"{instance_id}\t{answer}\n")
            fa_txt.flush()

            json_answer = {"instance_id":instance_id, "answer":answer}
            json_data.append(json_answer)
        json.dump(json_data, fw_json, indent=4)