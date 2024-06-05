from common import directories, logging_config, constants
from data_loader.base_data_loader import BaseDataLoader
from data_loader.fb_data_loader import FacebookDataLoader
from data_loader.tw_data_loader import TwitterDataLoader
from data_loader.news_data_loader import NewsDataLoader
from data_loader.fb_covid_data_loader import CovidFacebookDataLoader
from data_loader.ig_covid_data_loader import InstagramDataLoader

logger = logging_config.setup_logger(constants.LOGGER_NAME)


def chatgpt_data_loader():
    platforms = ["facebook", "news", "instagram", "youtube", "reddit", "twitter"]
    for platform in platforms:
        logger.info(f"Loading data for {platform}")
        if platform == "twitter":
            file_path = directories.RAW_TWIITER_DIR
            loader = TwitterDataLoader(
                file_path=file_path, platform=platform, topic="chatgpt"
            )
        else:
            file_path = f"{directories.RAW_GPT_DIR}/{platform}.csv"
            if platform == "facebook":
                loader = FacebookDataLoader(
                    file_path=file_path, platform=platform, topic="chatgpt"
                )
            elif platform == "news":
                loader = NewsDataLoader(
                    file_path=file_path, platform=platform, topic="chatgpt"
                )
            else:
                loader = BaseDataLoader(
                    file_path=file_path, platform=platform, topic="chatgpt"
                )

        loader.process_data()
        logger.info("-" * 100)


def covid_data_loader():
    platforms = ["news", "youtube", "instagram", "twitter", "facebook"]

    for platform in platforms:
        if platform == "twitter":
            file_path = directories.RAW_COVID_TWITTER_DIR
            loader = TwitterDataLoader(
                file_path=file_path, platform=platform, topic="covid"
            )
        elif platform == "facebook":
            file_path = directories.RAW_COVID_FACEBOOK_DIR
            loader = CovidFacebookDataLoader(
                file_path=file_path, platform=platform, topic="covid"
            )
        elif platform == "instagram":
            file_path = directories.RAW_COVID_INSTAGRAM_DIR
            loader = InstagramDataLoader(
                file_path=file_path, platform=platform, topic="covid"
            )
        else:
            file_path = f"{directories.RAW_COVID_DIR}/{platform}.csv"
            if platform == "news":
                loader = NewsDataLoader(
                    file_path=file_path, platform=platform, topic="covid"
                )
            else:
                loader = BaseDataLoader(
                    file_path=file_path, platform=platform, topic="covid"
                )
        loader.process_data()
        logger.info("-" * 100)


if __name__ == "__main__":
    logger.info("*" * 100)
    logger.info(f"{'*' * 43} CHATGPT DATA {'*' * 43}")
    logger.info("*" * 100)
    chatgpt_data_loader()
    logger.info("*" * 100)
    logger.info(f"{'*' * 44} COVID DATA {'*' * 44}")
    logger.info("*" * 100)
    covid_data_loader()
