import os
import ast
import json
import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm


def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = dd.read_csv(csv_path, converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'}).head(n=nrows)
    print(df.shape)
    for column in JSON_COLUMNS:
        df[column] = df[column].apply(ast.literal_eval)

    for column in tqdm(JSON_COLUMNS):
        column_as_df = pd.json_normalize(df[column])
        column_as_df.columns = ["{0}.{1}".format(column, subcolumn) for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    unique_only_columns = [column for column in df.columns if len(df[column].unique()) == 1]

    print(f"Columns with a single unique value: {unique_only_columns}")

    for column in unique_only_columns:
        df = df.drop(column, axis=1)

    print("\nColumns after dropping those with only 1 unique value:")
    print(df.columns)

    dt = (df.isnull().sum() / len(df.index)) * 100
    missing_values = pd.DataFrame({'Columns': dt.index, 'Null Values Count': dt.values})

    print("\nPercentage of missing values in each column:")
    print(missing_values)

    for i in missing_values.index:
        column_name = missing_values.iloc[i]['Columns']
        null_percentage = missing_values.iloc[i]['Null Values Count']
        if null_percentage > 90 and column_name != 'totals.transactionRevenue':
            df = df.drop(missing_values.iloc[i]['Columns'], axis=1)

    print("\nColumns after dropping those with more than 90% missing values:")
    print(df.columns)

    df['totals.bounces'] = df['totals.bounces'].fillna(0)
    df['totals.newVisits'] = df['totals.newVisits'].fillna(0)
    df['totals.hits'] = df['totals.hits'].fillna(0)
    df['visitNumber'] = df['visitNumber'].fillna(0)
    df = df.astype(
        {"totals.bounces": 'int64', "totals.newVisits": 'int64', 'totals.hits': 'int64', 'visitNumber': 'int64'})

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


def preprocess_data(train_df, test_df):
    train_df['totals.transactionRevenue'] = pd.to_numeric(
        train_df['totals.transactionRevenue'].str.replace('[\$,]', '', regex=True), errors='coerce')
    train_df.dropna(subset=['totals.transactionRevenue'], inplace=True)

    return train_df, test_df


def transform_target_variable(train_df):
    train_df['log_transactionRevenue'] = np.log1p(train_df['totals.transactionRevenue'])
    return train_df
