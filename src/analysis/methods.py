import pandas as pd
import os

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