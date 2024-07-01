import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(train_df):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 18))

    # Distribution of channelGrouping
    channel_counts = train_df['channelGrouping'].value_counts()
    sns.barplot(x=channel_counts.index, y=channel_counts.values, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Distribution of channelGrouping')
    axes[0, 0].set_xlabel('Channel Grouping')
    axes[0, 0].set_ylabel('Count')

    # Distribution of Visits by Browser
    browser_counts = train_df['device.browser'].value_counts().head(10)
    sns.barplot(x=browser_counts.values, y=browser_counts.index, orient='h', ax=axes[0, 1], palette='plasma')
    axes[0, 1].set_title('Top 10 Browsers by Number of Visits')
    axes[0, 1].set_xlabel('Number of Visits')
    axes[0, 1].set_ylabel('Browser')

    # Distribution of Visits by Operating System
    os_counts = train_df['device.operatingSystem'].value_counts().head(10)
    sns.barplot(x=os_counts.values, y=os_counts.index, orient='h', ax=axes[1, 0], palette='magma')
    axes[1, 0].set_title('Top 10 Operating Systems by Number of Visits')
    axes[1, 0].set_xlabel('Number of Visits')
    axes[1, 0].set_ylabel('Operating System')

    # Distribution of Visits by Mobile vs Non-Mobile
    mobile_counts = train_df['device.isMobile'].value_counts()
    sns.barplot(x=mobile_counts.index, y=mobile_counts.values, ax=axes[1, 1], palette='rocket')
    axes[1, 1].set_title('Distribution of Visits by Mobile vs Non-Mobile')
    axes[1, 1].set_xlabel('Is Mobile Device')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('visualization1.png')
    plt.show()
