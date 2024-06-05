# questo file mi serve
DB_CONFIG = "/home/Desktop/viral_conversations/keys/db.json"

LOG_DIR = "/home/Desktop/viral_conversations/logs/"

PLOT_DIR = "/home/Desktop/viral_conversations/figures"

GENERAL_DIR = "/media/Volume/chatgpt"

# Raw data directories
RAW_DIR = f"{GENERAL_DIR}/raw_data"
GDELT_FILE = f"{GENERAL_DIR}/clean_data/gdelt_news.csv"
PLATFORM_NEWS_DIR = f"{GENERAL_DIR}/platform_news"


# ChatGPT raw data directories
RAW_GPT_DIR = f"{RAW_DIR}/chatgpt"
RAW_TWIITER_DIR = f"{RAW_GPT_DIR}/twitter/all_twitter_cleaned_timelines"

# COVID raw data directories
RAW_COVID_DIR = f"{RAW_DIR}/covid"
RAW_COVID_TWITTER_DIR = f"{RAW_COVID_DIR}/twitter"
RAW_COVID_FACEBOOK_DIR = f"{RAW_COVID_DIR}/facebook"
RAW_COVID_INSTAGRAM_DIR = f"{RAW_COVID_DIR}/instagram"
REDDIT_DUMP_DIR = f"{RAW_COVID_DIR}/reddit_dump"


# topic modeling directories
TOPIC_MODELS_DIR = f"{GENERAL_DIR}/topic_modeling"
