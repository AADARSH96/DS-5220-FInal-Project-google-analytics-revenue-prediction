import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def plot_distributions(train_df):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3, 2, figsize=(18, 24))

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
    plt.savefig('distributions1.png')

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))

    # Distribution of Visits by Continent (Counts)
    continent_counts = train_df['geoNetwork.continent'].value_counts()
    sns.barplot(x=continent_counts.values, y=continent_counts.index, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Distribution of Visits by Continent (Counts)')
    axes[0, 0].set_xlabel('Continent')
    axes[0, 0].set_ylabel('Number of Visits')

    # Distribution of Visits by Continent (Mean Transaction Revenue)
    revenue_by_continent = train_df.groupby('geoNetwork.continent')['totals.transactionRevenue'].mean().sort_values(
        ascending=False)
    sns.barplot(x=revenue_by_continent.values, y=revenue_by_continent.index, ax=axes[0, 1], palette='plasma')
    axes[0, 1].set_title('Mean Transaction Revenue by Continent')
    axes[0, 1].set_xlabel('Mean Transaction Revenue')
    axes[0, 1].set_ylabel('Continent')

    # Top 10 Countries by Number of Visits
    top_countries = train_df['geoNetwork.country'].value_counts().head(10)
    sns.barplot(x=top_countries.values, y=top_countries.index, orient='h', ax=axes[1, 0], palette='magma')
    axes[1, 0].set_title('Top 10 Countries by Number of Visits')
    axes[1, 0].set_xlabel('Number of Visits')
    axes[1, 0].set_ylabel('Country')

    # Top 10 Countries by Mean Transaction Revenue
    revenue_by_country = train_df.groupby('geoNetwork.country')['totals.transactionRevenue'].mean().sort_values(
        ascending=False).head(10)
    sns.barplot(x=revenue_by_country.values, y=revenue_by_country.index, orient='h', ax=axes[1, 1], palette='rocket')
    axes[1, 1].set_title('Top 10 Countries by Mean Transaction Revenue')
    axes[1, 1].set_xlabel('Mean Transaction Revenue')
    axes[1, 1].set_ylabel('Country')

    # Top 10 Cities by Number of Visits
    top_cities = train_df['geoNetwork.city'].value_counts().head(10)
    sns.barplot(x=top_cities.values, y=top_cities.index, orient='h', ax=axes[2, 0], palette='viridis')
    axes[2, 0].set_title('Top 10 Cities by Number of Visits')
    axes[2, 0].set_xlabel('Number of Visits')
    axes[2, 0].set_ylabel('City')

    # Top 10 Cities by Mean Transaction Revenue
    revenue_by_city = train_df.groupby('geoNetwork.city')['totals.transactionRevenue'].mean().sort_values(
        ascending=False).head(10)
    sns.barplot(x=revenue_by_city.values, y=revenue_by_city.index, orient='h', ax=axes[2, 1], palette='plasma')
    axes[2, 1].set_title('Top 10 Cities by Mean Transaction Revenue')
    axes[2, 1].set_xlabel('Mean Transaction Revenue')
    axes[2, 1].set_ylabel('City')

    # Adjust layout
    plt.tight_layout()
    plt.savefig('distributions2.png')

    fig, axes = plt.subplots(2, 2, figsize=(18, 15))

    # Distribution of trafficSource.campaign
    campaign_counts = train_df['trafficSource.campaign'].value_counts().head(10)
    sns.barplot(x=campaign_counts.values, y=campaign_counts.index, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Top 10 Campaigns by Number of Visits')
    axes[0, 0].set_xlabel('Number of Visits')
    axes[0, 0].set_ylabel('Campaign')

    # Distribution of trafficSource.source
    source_counts = train_df['trafficSource.source'].value_counts().head(10)
    sns.barplot(x=source_counts.values, y=source_counts.index, ax=axes[0, 1], palette='plasma')
    axes[0, 1].set_title('Top 10 Traffic Sources by Number of Visits')
    axes[0, 1].set_xlabel('Number of Visits')
    axes[0, 1].set_ylabel('Traffic Source')

    # Distribution of trafficSource.medium
    medium_counts = train_df['trafficSource.medium'].value_counts().head(10)
    sns.barplot(x=medium_counts.values, y=medium_counts.index, ax=axes[1, 0], palette='magma')
    axes[1, 0].set_title('Top 10 Traffic Mediums by Number of Visits')
    axes[1, 0].set_xlabel('Number of Visits')
    axes[1, 0].set_ylabel('Traffic Medium')

    # Distribution of trafficSource.keyword
    keyword_counts = train_df['trafficSource.keyword'].value_counts().head(10)
    sns.barplot(x=keyword_counts.values, y=keyword_counts.index, ax=axes[1, 1], palette='rocket')
    axes[1, 1].set_title('Top 10 Keywords by Number of Visits')
    axes[1, 1].set_xlabel('Number of Visits')
    axes[1, 1].set_ylabel('Keyword')

    # Adjust layout
    plt.tight_layout()
    plt.savefig('distributions3.png')


def process_time(posix_time):
    return datetime.utcfromtimestamp(posix_time).strftime('%Y-%m-%d %H:%M:%S')


def plot_trends(df):
    # Ensure 'date' and 'visitStartTime' columns are in the correct format
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
    df['visitStartTime'] = df['visitStartTime'].apply(process_time)
    df['visitHour'] = df['visitStartTime'].apply(lambda x: int(x[11:13]))

    # Group by date and compute daily trends
    df_date = df.groupby('date').size()

    # Group by month and compute monthly trends
    df_monthly_counts = df.groupby(pd.Grouper(key='date', freq='M')).size()

    # Hourly trends
    hourly_counts = df['visitHour'].value_counts().sort_index()

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(20, 18))

    # Date vs Number of Visits
    axs[0].plot(df_date.index, df_date.values, label='Number of Visits')
    axs[0].set_title('Date vs Number of Visits')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Number of Visits')
    axs[0].grid(ls='--')

    # Monthly Number of Visits
    months = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    month_labels = [f"{months[i.month]}_{i.year}" for i in df_monthly_counts.index]
    axs[1].bar(month_labels, df_monthly_counts, color='skyblue')
    for i, count in enumerate(df_monthly_counts):
        axs[1].text(i, count + 1000, f"{round(100 * (count / df_monthly_counts.sum()), 2)}%", size=12, rotation=90)
    axs[1].set_title('Month vs Number of Visits')
    axs[1].set_ylabel('Number of Visits')
    axs[1].grid(ls='--')
    axs[1].tick_params(axis='x', rotation=45)

    # Hourly Number of Visits
    axs[2].bar(hourly_counts.index, hourly_counts.values, color='lightgreen')
    for i, count in hourly_counts.items():
        axs[2].text(i, count + 2000, f"{round(100 * (count / hourly_counts.sum()), 2)}%", size=12, rotation=90)
    axs[2].set_title('Visit Hour vs Number of Visits')
    axs[2].set_xlabel('Visit Hour')
    axs[2].set_ylabel('Number of Visits')
    axs[2].grid(ls='--')
    axs[2].set_xticks(range(24))

    plt.tight_layout()
    plt.savefig('combined_trends.png')
