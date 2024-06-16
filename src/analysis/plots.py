import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import seaborn as sns

##############################################################################
PLATFORM_DISPLAY_NAMES = {
    "fb": "Facebook",
    "ig": "Instagram",
}

TOPIC_DISPLAY_NAMES = {
    "gpt3": "GPT 3.5",
    "gpt4": "GPT 4",
    "apple": "Apple Vision Pro"
}

SENTIMENT_DISPLAY_NAMES = {
    "positive": "POSITIVE",
    "neutral": "NEUTRAL",
    "negative": "NEGATIVE"
}

def _assign_platform_colors(color_palette):
        platform_list = list(PLATFORM_DISPLAY_NAMES.values())
        return {platform: color for platform, color in zip(platform_list, color_palette)}

def _replace_platform_names(data):
        """Replace platform names with their display names."""
        data.loc[:, "platform"] = data["platform"].replace(PLATFORM_DISPLAY_NAMES)
        return data

platform_colors = _assign_platform_colors(sns.color_palette("muted", 2))
##############################################################################

## QUANTITATIVE ANALYSIS

def plot_daily_posts(line_data, ax, topic):
    line_data = _replace_platform_names(line_data)
    ax = sns.lineplot(data=line_data, x="date", y="cumulative_posts", hue="platform", palette=platform_colors, \
        linewidth=4, alpha=0.7, ax=ax,
    )

    ax.tick_params(axis="both", which="major", labelsize=22)
    num_ticks = 6
    x_dates = sorted(
        line_data["date"].unique().tolist()
    )  # Ensuring the dates are sorted
    spacing = len(x_dates) // (num_ticks - 1)
    selected_ticks = [x_dates[0]]
    for i in range(1, num_ticks - 1):
        selected_ticks.append(x_dates[i * spacing])
    selected_ticks.append(x_dates[-1])
    ax.set_xticks(selected_ticks)
    plt.setp(ax.get_xticklabels(), rotation=30)
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Number of Posts", fontsize=22)
    ax.set(yscale="log")
    new_topic_name = {"gpt3":"GPT-3.5", "gpt4":"GPT-4", "apple":"Apple Vision Pro"}
    ax.text(0.02, 0.98, new_topic_name[topic], fontsize=20, va="top", transform=ax.transAxes,)

    handles, _ = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_linewidth(4)
    ax.legend(loc="lower right", fontsize=22, markerscale=2, handles=handles)


def plot_interaction_dist(scatter_data, ax, topic):
    scatter_data = _replace_platform_names(scatter_data)
    ax = sns.scatterplot(data=scatter_data, x="interaction", y="post_count", hue="platform", \
        palette=platform_colors, s=80, alpha=0.5, linewidth=0.0, ax=ax,
    )

    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.set_xlabel("Number of Interactions", fontsize=22)
    ax.set_ylabel("Number of Posts", fontsize=22)
    ax.set(xscale="log", yscale="log")
    new_topic_name = {"gpt3":"GPT-3.5", "gpt4":"GPT-4", "apple":"Apple Vision Pro"}
    ax.text(0.02, 0.98, new_topic_name[topic], fontsize=20, va="top", transform=ax.transAxes,)

    handles, _ = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_linewidth(4)
    ax.legend(loc="upper right", fontsize=22, markerscale=2, handles=handles,)


def logistic_model_plot(x_col="day", y_col="cumulative_unique_users", ax=None,
        gpt3_data=None, gpt3_sigmoid_fit=None,
        gpt4_data=None, gpt4_sigmoid_fit=None, 
        apple_data=None, apple_sigmoid_fit=None
        ):
        gpt3_data = gpt3_data.copy()
        gpt3_data = _replace_platform_names(gpt3_data)
        platform_display_name = gpt3_data.iloc[0]["platform"]
        # set figure title, x and y labels
        ax.set_title(platform_display_name, fontsize=24,)
        ax.set_xlabel("Days", fontsize=22)
        ax.set_ylabel("Cumulative Number of Users", fontsize=22)
        # set x and y ticks
        ax.tick_params(axis="both", which="major", labelsize=22)
        
        if gpt3_data is not None:
            # GPT3 DATA
            # plot the data
            ax.plot(
                gpt3_data[x_col],
                gpt3_data[y_col],
                linewidth=4,
                color=platform_colors[platform_display_name],
                alpha=0.8,
                label="GPT-3.5 Data",
            )
            # plot the fitted sigmoid
            ax.plot(
                gpt3_data[x_col],
                gpt3_sigmoid_fit,
                linewidth=4,
                color="black",
                alpha=0.6,
                label="GPT-3.5 Fit",
            ) 
        
        if gpt4_data is not None:
            # GPT4 DATA
            # plot the data
            ax.plot(
                gpt4_data[x_col],
                gpt4_data[y_col],
                linewidth=4,
                linestyle="dashed",
                color=platform_colors[platform_display_name],
                alpha=0.8,
                label="GPT-4 Data",
            )
            # plot the fitted sigmoid
            ax.plot(
                gpt4_data[x_col],
                gpt4_sigmoid_fit,
                linewidth=4,
                linestyle="dashed",
                color="black",
                alpha=0.6,
                label="GPT-4 Fit",
            )
        
        if apple_data is not None:
            # APPLE DATA
            # plot the data
            ax.plot(
                apple_data[x_col],
                apple_data[y_col],
                linewidth=4,
                linestyle="dotted",
                color=platform_colors[platform_display_name],
                alpha=0.8,
                label="Apple Vision Pro Data",
            )

            # plot the fitted sigmoid
            ax.plot(
                apple_data[x_col],
                apple_sigmoid_fit,
                linewidth=4,
                linestyle="dotted",
                color="black",
                alpha=0.6,
                label="Apple Vision Pro Fit",
            )

        # set legend
        ax.legend(loc="best", fontsize=22)

        return ax

# class for fitting data to sigmoid function    
class LogisticModel:
    def __init__(self, platforms):
        self.platforms = platforms

    def _process_data(self, df):
        # instead of dates (i.e. 2020-01-01), we want to use the number of days since the first date (i.e. 0, 1, 2, ...)
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = (df["date"] - df["date"].min()).dt.days
        df["day"] = df["day"].astype(int)
        # normalize the cumulative unique users (MinMax scaling)
        df["cumulative_unique_users"] = (
            df["cumulative_unique_users"] - df["cumulative_unique_users"].min()
        ) / (df["cumulative_unique_users"].max() - df["cumulative_unique_users"].min())
        return df

    def _sigmoid(self, t, alpha, beta):
        return 1 / (1 + np.exp(-alpha * (t - beta)))

    def _fit_sigmoid(self, platform_df, platform_name):
        try:
            popt, _ = curve_fit(self._sigmoid, platform_df["day"], platform_df["cumulative_unique_users"], maxfev=10000, p0=None)
            residuals = platform_df["cumulative_unique_users"] - self._sigmoid(platform_df["day"], *popt)
            rmse = np.sqrt( mean_squared_error(platform_df["cumulative_unique_users"], self._sigmoid(platform_df["day"], *popt),) )
            return round(popt[0], 3), round(popt[1], 3), popt, round(np.sum(residuals**2), 3), round(rmse, 3)
        
        except Exception as e:
            print(f"Error fitting sigmoid for {platform_name}: {e}")

    def _AUC_sigmoid(self, platform_df, popt):
        T = platform_df["day"].max()
        area, _ = quad(self._sigmoid, 0, T, args=(popt[0], popt[1]))
        return round(area / T, 3)
    
    def run(self):
        daily_user_counts_file = os.path.join(f"../../data/quantitative_analysis", f'daily_user_counts.csv')
        data = pd.read_csv(daily_user_counts_file)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        for idx, platform in enumerate(self.platforms):
            ax = axs[idx]

            subset = data[data["platform"] == platform]
            gpt3_subset = subset[subset["topic"] == "gpt3"]
            gpt4_subset = subset[subset["topic"] == "gpt4"]
            apple_subset = subset[subset["topic"] == "apple"]

            gpt3_subset = self._process_data(gpt3_subset)
            gpt4_subset = self._process_data(gpt4_subset)
            apple_subset = self._process_data(apple_subset)

            # Fit the sigmoid function
            gpt3_alpha, gpt3_beta, gpt3_popt, gpt3_residuals, gpt3_rmse = self._fit_sigmoid(gpt3_subset, platform)
            gpt4_alpha, gpt4_beta, gpt4_popt, gpt4_residuals, gpt4_rmse = self._fit_sigmoid(gpt4_subset, platform)
            apple_alpha, apple_beta, apple_popt, apple_residuals, apple_rmse = self._fit_sigmoid(apple_subset, platform)
            print(
                f"GPT3: alpha: {gpt3_alpha}, beta: {gpt3_beta}, residuals: {gpt3_residuals}, rmse: {gpt3_rmse}"
            )
            print(
                f"GPT4: alpha: {gpt4_alpha}, beta: {gpt4_beta}, residuals: {gpt4_residuals}, rmse: {gpt4_rmse}"
            )
            print(
                f"APPLE: alpha: {apple_alpha}, beta: {apple_beta}, residuals: {apple_residuals}, rmse: {apple_rmse}"
            )

            # calculate the AUC
            gpt3auc = self._AUC_sigmoid(gpt3_subset, gpt3_popt)
            gpt4_auc = self._AUC_sigmoid(gpt4_subset, gpt4_popt)
            apple_auc = self._AUC_sigmoid(apple_subset, apple_popt)
            print(f"GPT3: auc: {gpt3auc}, GPT4: auc: {gpt4_auc}, APPLE: auc: {apple_auc}")

            # plot the sigmoid function
            ax = logistic_model_plot(
                x_col="day",
                y_col="cumulative_unique_users",
                ax=ax,
                gpt3_data=gpt3_subset, gpt3_sigmoid_fit=self._sigmoid(gpt3_subset["day"], *gpt3_popt),
                gpt4_data=gpt4_subset, gpt4_sigmoid_fit=self._sigmoid(gpt4_subset["day"], *gpt4_popt), 
                apple_data=apple_subset, apple_sigmoid_fit=self._sigmoid(apple_subset["day"], *apple_popt)
            )
            print(f"Finished plotting {platform}")
            print("-" * 100)
        fig.tight_layout()
        

## SENTIMENT ANALYSIS

def reactions_pie_chart(topic, ax):
    titles = {"gpt3" : f'GPT 3.5 Facebook Reactions', "gpt4" : f'GPT 4 Facebook Reactions', "apple" : f'Apple Vision Pro Facebook Reactions'}
    data = pd.read_csv(f"../../data/clean/fb_{topic}.csv")
    
    columns_of_interest = ['Love', 'Wow', 'Haha', 'Sad', 'Angry', 'Care']
    values = data[columns_of_interest].sum()
    
    # Plot the pie chart on the corresponding subplot
    ax.pie(values, labels=columns_of_interest, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    ax.set_title(titles[topic])

def emotions_bar_plot(emotion_data):
    emotion_data.loc[:, "topic"] = emotion_data["topic"].replace(TOPIC_DISPLAY_NAMES)
    
    def normalize_data(data):
        # calculate the counts of each emotion for each topic
        emotion_counts = data.groupby(['topic', 'emotion']).size().reset_index(name='count')
        # calculate the total counts for each topic
        total_counts = data['topic'].value_counts().reset_index()
        total_counts.columns = ['topic', 'total_count']
        # merge the emotion counts with the total counts
        emotion_counts = emotion_counts.merge(total_counts, on='topic')
        # normalize the counts by dividing by the total counts
        emotion_counts['normalized_count'] = emotion_counts['count'] / emotion_counts['total_count']
        return emotion_counts
    # normalize data
    fb_data = normalize_data(emotion_data[emotion_data['platform'] == 'fb'])
    insta_data = normalize_data(emotion_data[emotion_data['platform'] == 'ig'])

    _, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    # plot for Facebook
    sns.barplot(x='topic', y='normalized_count', hue='emotion', data=fb_data, palette='pastel', ax=axes[0])
    axes[0].set_title('Facebook')
    axes[0].set_xlabel('Topic')
    axes[0].set_ylabel('')
    axes[0].legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plot for Instagram
    sns.barplot(x='topic', y='normalized_count', hue='emotion', data=insta_data, palette='pastel', ax=axes[1])
    axes[1].set_title('Instagram')
    axes[1].set_xlabel('Topic')
    axes[1].set_ylabel('')
    axes[1].legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def sentiments_conf_matrix(sentiment_data):
    sentiment_data.loc[:, "topic"] = sentiment_data["topic"].replace(TOPIC_DISPLAY_NAMES)
    sentiment_data.loc[:, "sentiment"] = sentiment_data["sentiment"].replace(SENTIMENT_DISPLAY_NAMES)

    def create_confusion_matrix(df, platform):
        platform_df = df[df['platform'] == platform]
        confusion_df = pd.crosstab(platform_df['topic'], platform_df['sentiment'], normalize='index')
        return confusion_df

    # create confusion matrices for Facebook and Instagram
    fb_confusion_matrix = create_confusion_matrix(sentiment_data, 'fb')
    insta_confusion_matrix = create_confusion_matrix(sentiment_data, 'ig')

    _, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    # plot for Facebook
    sns.heatmap(fb_confusion_matrix, annot=True, cmap='Greens', ax=axes[0])
    axes[0].set_title('Facebook')
    axes[0].set_xlabel('Sentiment')
    axes[1].set_ylabel('Topic')
    for text in axes[0].texts: text.set_fontsize(16)
    # plot for Instagram
    sns.heatmap(insta_confusion_matrix, annot=True, cmap='Greens', ax=axes[1])
    axes[1].set_title('Instagram')
    axes[1].set_xlabel('Sentiment')
    axes[1].set_ylabel('Topic')
    for text in axes[1].texts: text.set_fontsize(16)

    plt.tight_layout()
    plt.show()

# here we generate two kind of plots
def plot_LLM_conversations(sentiment_data):
    
    ## CONFUSION MATRIX
    
    sentiment_data.loc[:, "platform"] = sentiment_data["platform"].replace(PLATFORM_DISPLAY_NAMES)
    sentiment_data = sentiment_data.dropna(subset=['category'])

    def create_confusion_matrix(df, topic):
        topic_df = df[df['topic'] == topic]
        confusion_df = pd.crosstab(topic_df['category'], topic_df['platform'], normalize='index')
        return confusion_df
    
    topics = ['gpt3', 'gpt4']
    confusion_matrices = {topic: create_confusion_matrix(sentiment_data, topic) for topic in topics}
    _, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    for i, topic in enumerate(topics):
        sns.heatmap(confusion_matrices[topic], annot=True, cmap='Blues', ax=axes[i])
        title = "GPT 3.5" if  topic=='gpt3' else "GPT 4"
        axes[i].set_title(title)
        axes[i].set_xlabel('Platform')
        axes[i].set_ylabel('Category')
        for text in axes[i].texts: text.set_fontsize(16)

    plt.tight_layout()
    plt.show()
    
    
    ## VIOLIN PLOT

    sentiment_data.loc[:, "sentiment"] = sentiment_data["sentiment"].replace(SENTIMENT_DISPLAY_NAMES)
    category_order = sorted(sentiment_data['category'].unique())
    sns.set_palette('pastel')
    _, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    platforms = sentiment_data['platform'].unique()
    for ax, platform in zip(axes, platforms):
        platform_data = sentiment_data[sentiment_data['platform'] == platform]
        
        sns.violinplot(data=platform_data, x='category', y='sentiment',  ax=ax, split=True, order=category_order)
        ax.set_title(platform)
        ax.set_ylabel('')
        ax.set_xlabel('ChatGPT Conversations Categories')