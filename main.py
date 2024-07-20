import dask.dataframe as dd
from src.preprocessing import load_df, preprocess_data, group_features
from src.modeling import evaluate_models
from src.vizualization import plot_distributions, plot_trends
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import warnings  # For managing warnings
warnings.simplefilter(action='ignore')  # Ignore future warnings
import numpy as np


if __name__ == "__main__":
    # Load data
    data_1 = load_df('/Users/aadarsh/study/DS 5220/project/train.csv')
    data_2 = load_df('/Users/aadarsh/study/DS 5220/project/train_df.csv', json_cols=[])
    df = dd.concat([data_1, data_2]).drop_duplicates()
    df = df.compute()

    # run visualization

    df = preprocess_data(df)
    grouped_df = group_features(df)
    grouped_df['log_transactionRevenue'] = np.log1p(grouped_df['revenue_sum'])

    # Split data into train and validation sets
    X = grouped_df.drop(columns=['revenue_sum', 'log_transactionRevenue', 'fullVisitorId'])
    y = grouped_df['log_transactionRevenue']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate models
    results = evaluate_models(X_train, X_val, y_train, y_val)
    print(f"{'*' * 50} Model Scores {'*' * 50}")
    print(tabulate(results, headers='keys', tablefmt='psql'))