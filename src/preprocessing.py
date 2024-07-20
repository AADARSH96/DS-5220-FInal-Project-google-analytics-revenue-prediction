import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm
import category_encoders as ce
import json
import ast
from src.vizualization import plot_distributions,plot_trends


def load_df(csv_path, nrows=None, json_cols=['device', 'geoNetwork', 'totals', 'trafficSource']):
    print(f"Loading data from CSV... {csv_path}")

    # Load data into Dask DataFrame
    if json_cols:
        df = dd.read_csv(csv_path, converters={col: json.loads for col in json_cols},
                         dtype={'fullVisitorId': 'str'}).head(n=nrows)
        print("Normalizing JSON columns...")
        # Normalize JSON columns
        for col in tqdm(json_cols, desc="Processing JSON columns"):
            df[col] = df[col].apply(ast.literal_eval)
            col_df = pd.json_normalize(df[col])
            col_df.columns = ["{0}.{1}".format(col, subcol) for subcol in col_df.columns]
            df = df.drop(col, axis=1).merge(col_df, right_index=True, left_index=True)
    else:
        dtype = {
            'totals.bounces': 'float64',
            'totals.pageviews': 'float64',
            'trafficSource.adContent': 'object',
            'trafficSource.adwordsClickInfo.adNetworkType': 'object',
            'trafficSource.adwordsClickInfo.gclId': 'object',
            'trafficSource.adwordsClickInfo.slot': 'object',
            'trafficSource.referralPath': 'object',
            'fullVisitorId': 'str'
        }
        df = dd.read_csv(csv_path, dtype=dtype).head(n=nrows)

    print("Identifying columns with a single unique value...")
    unique_only_columns = []
    # Identifying columns with a single unique value
    for column in list(df.columns):
        if len(list(df[column].unique())) == 1:
            unique_only_columns.append(column)

    print(f"Columns with a single unique value: {unique_only_columns}")

    # Dropping columns with a single unique value
    for column in unique_only_columns:
        df = df.drop(column, axis=1)

    print("Calculating percentage of missing values in each column...")
    # Calculating percentage of missing values in each column
    dt = (df.isnull().sum() / len(df.index)) * 100
    missing_values = pd.DataFrame({'Columns': dt.index, 'Null Values Count': dt.values})

    print("Dropping columns with more than 90% missing values (except 'totals.transactionRevenue')...")
    # Dropping columns with more than 90% missing values, except for 'totals.transactionRevenue'
    for i in missing_values.index:
        column_name = missing_values.iloc[i]['Columns']
        null_percentage = missing_values.iloc[i]['Null Values Count']
        if null_percentage > 90 and column_name != 'totals.transactionRevenue':
            df = df.drop(missing_values.iloc[i]['Columns'], axis=1)

    print("Handling missing values and setting data types...")
    # Handle missing values and data types
    fill_na_cols = ['totals.bounces', 'totals.newVisits', 'totals.pageviews', 'totals.hits', 'visitNumber']
    df[fill_na_cols] = df[fill_na_cols].fillna(0).astype('int64')
    df['trafficSource.isTrueDirect'].fillna('False', inplace=True)
    df['date'] = pd.to_datetime(df["date"], infer_datetime_format=True, format="%Y%m%d")
    df['weekday'] = df.date.dt.weekday
    df['day'] = df.date.dt.day
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    # Select relevant columns
    relevant_cols = ['channelGrouping', 'date', 'fullVisitorId', 'sessionId', 'visitId',
                     'visitNumber', 'visitStartTime', 'device.browser', 'device.operatingSystem',
                     'device.isMobile', 'device.deviceCategory', 'geoNetwork.continent',
                     'geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.region',
                     'geoNetwork.metro', 'geoNetwork.city', 'geoNetwork.networkDomain',
                     'totals.hits', 'totals.pageviews', 'totals.bounces', 'totals.newVisits',
                     'totals.transactionRevenue', 'trafficSource.campaign', 'trafficSource.source',
                     'trafficSource.medium', 'trafficSource.keyword', 'trafficSource.isTrueDirect',
                     'trafficSource.referralPath', 'weekday', 'day', 'month', 'year']

    print("Selecting relevant columns...")
    return df[relevant_cols]  # Compute Dask DataFrame into pandas DataFrame


def preprocess_data(df):
    print("Preprocessing data...")
    df['totals.transactionRevenue'] = pd.to_numeric(
        df['totals.transactionRevenue'].str.replace('[\$,]', '', regex=True), errors='coerce')
    df['totals.transactionRevenue'].fillna(0, inplace=True)
    pyarrow_string_cols = df.select_dtypes(include=['string[pyarrow]']).columns
    df[pyarrow_string_cols] = df[pyarrow_string_cols].astype('object')

    plot_distributions(df)
    plot_trends(df)

    print("Encoding categorical columns...")
    # Encode categorical columns
    categorical_cols = ['channelGrouping', 'device.browser', 'device.operatingSystem', 'device.isMobile',
                        'device.deviceCategory', 'geoNetwork.continent', 'geoNetwork.subContinent',
                        'geoNetwork.country', 'geoNetwork.region', 'geoNetwork.metro', 'geoNetwork.city',
                        'geoNetwork.networkDomain', 'totals.bounces', 'totals.newVisits', 'trafficSource.campaign',
                        'trafficSource.source', 'trafficSource.medium', 'trafficSource.keyword',
                        'trafficSource.isTrueDirect', 'trafficSource.referralPath', 'weekday', 'day', 'month', 'year']

    encoder = ce.TargetEncoder(cols=categorical_cols, handle_missing='median')
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df['totals.transactionRevenue'])

    print("Filling missing values...")
    # Fill missing values
    df['totals.pageviews'].fillna(np.nanmedian(df['totals.pageviews']), inplace=True)

    print("Preprocessing complete.")
    return df


def group_features(df):
    print("Grouping features by fullVisitorId...")
    # Group by fullVisitorId and aggregate features
    grouped_df = df.groupby('fullVisitorId').agg({
        'totals.pageviews': [
            ('total_pageviews_max', lambda x: x.max()),
            ('total_pageviews_min', lambda x: x.min()),
            ('total_pageviews_mean', lambda x: x.mean()),
            ('total_pageviews_mode', lambda x: x.mode().iloc[0])
        ],
        'channelGrouping': [
            ('channelGrouping_max', lambda x: x.max()),
            ('channelGrouping_min', lambda x: x.min()),
            ('channelGrouping_mode', lambda x: x.mode().iloc[0])
        ],
        'visitNumber': [
            ('visitNumber_max', lambda x: x.max()),
            ('visitNumber_mean', lambda x: x.mean()),
            ('visitNumber_min', lambda x: x.min())
        ],
        'device.browser': [
            ('device_browser_max', lambda x: x.max()),
            ('device_browser_min', lambda x: x.min()),
            ('device_browser_mode', lambda x: x.mode().iloc[0])
        ],
        'device.operatingSystem': [
            ('device_operatingSystem_max', lambda x: x.max()),
            ('device_operatingSystem_min', lambda x: x.min()),
            ('device_operatingSystem_mode', lambda x: x.mode().iloc[0])
        ],
        'device.isMobile': [
            ('device_isMobile_max', lambda x: x.max()),
            ('device_isMobile_min', lambda x: x.min())
        ],
        'device.deviceCategory': [
            ('device_deviceCategory_max', lambda x: x.max()),
            ('device_deviceCategory_min', lambda x: x.min()),
            ('device_deviceCategory_mode', lambda x: x.mode().iloc[0])
        ],
        'geoNetwork.continent': [
            ('geoNetwork_continent_max', lambda x: x.max()),
            ('geoNetwork_continent_min', lambda x: x.min())
        ],
        'geoNetwork.subContinent': [
            ('geoNetwork_subContinent_max', lambda x: x.max()),
            ('geoNetwork_subContinent_min', lambda x: x.min())
        ],
        'geoNetwork.country': [
            ('geoNetwork_country_max', lambda x: x.max()),
            ('geoNetwork_country_min', lambda x: x.min())
        ],
        'geoNetwork.region': [
            ('geoNetwork_region_max', lambda x: x.max()),
            ('geoNetwork_region_min', lambda x: x.min())
        ],
        'geoNetwork.metro': [
            ('geoNetwork_metro_max', lambda x: x.max()),
            ('geoNetwork_metro_min', lambda x: x.min())
        ],
        'geoNetwork.city': [
            ('geoNetwork_city_max', lambda x: x.max()),
            ('geoNetwork_city_min', lambda x: x.min())
        ],
        'geoNetwork.networkDomain': [
            ('geoNetwork_networkDomain_max', lambda x: x.max()),
            ('geoNetwork_networkDomain_min', lambda x: x.min()),
            ('geoNetwork_networkDomain_mode', lambda x: x.mode().iloc[0])
        ],
        'totals.bounces': [
            ('totals_bounces_max', lambda x: x.max()),
            ('totals_bounces_min', lambda x: x.min())
        ],
        'totals.newVisits': [
            ('totals_newVisits_max', lambda x: x.max()),
            ('totals_newVisits_min', lambda x: x.min())
        ],
        'trafficSource.campaign': [
            ('trafficSource_campaign_max', lambda x: x.max()),
            ('trafficSource_campaign_min', lambda x: x.min()),
            ('trafficSource_campaign_mode', lambda x: x.mode().iloc[0])
        ],
        'trafficSource.source': [
            ('trafficSource_source_max', lambda x: x.max()),
            ('trafficSource_source_min', lambda x: x.min()),
            ('trafficSource_source_mode', lambda x: x.mode().iloc[0])
        ],
        'trafficSource.medium': [
            ('trafficSource_medium_max', lambda x: x.max()),
            ('trafficSource_medium_min', lambda x: x.min()),
            ('trafficSource_medium_mode', lambda x: x.mode().iloc[0])
        ],
        'trafficSource.keyword': [
            ('trafficSource_keyword_max', lambda x: x.max()),
            ('trafficSource_keyword_min', lambda x: x.min()),
            ('trafficSource_keyword_mode', lambda x: x.mode().iloc[0])
        ],
        'trafficSource.isTrueDirect': [
            ('trafficSource_isTrueDirect_max', lambda x: x.max()),
            ('trafficSource_isTrueDirect_min', lambda x: x.min())
        ],
        'trafficSource.referralPath': [
            ('trafficSource_referralPath_max', lambda x: x.max()),
            ('trafficSource_referralPath_min', lambda x: x.min()),
            ('trafficSource_referralPath_mode', lambda x: x.mode().iloc[0])
        ],
        'weekday': [
            ('weekday_max', lambda x: x.max()),
            ('weekday_min', lambda x: x.min())
        ],
        'day': [
            ('day_max', lambda x: x.max()),
            ('day_min', lambda x: x.min())
        ],
        'month': [
            ('month_max', lambda x: x.max()),
            ('month_min', lambda x: x.min())
        ],
        'year': [
            ('year_max', lambda x: x.max()),
            ('year_min', lambda x: x.min())
        ],
        'totals.transactionRevenue': [('revenue_sum', lambda x: x.dropna().sum())]
    })
    grouped_df.columns = grouped_df.columns.droplevel()
    grouped_df = grouped_df.reset_index()
    print(grouped_df.columns)
    print("Grouping complete.")
    return grouped_df



