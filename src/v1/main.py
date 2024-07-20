from src.v1.preprocessing import load_df
from src.v1.preprocessing import preprocess_data, transform_target_variable
from src.v1.vizualization import plot_distributions
from src.v1.modeling import define_preprocessor, train_and_evaluate_models, print_results
from src.v1.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET


def main():
    train_csv_path = '../../data/train.csv'
    test_csv_path = '../../data/test.csv'

    train_df = load_df(train_csv_path)
    test_df = load_df(test_csv_path)

    train_df, test_df = preprocess_data(train_df, test_df)
    train_df = transform_target_variable(train_df)

    plot_distributions(train_df)

    X = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = train_df[TARGET]

    preprocessor = define_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    results_df = train_and_evaluate_models(X, y, preprocessor)

    print_results(results_df)


if __name__ == '__main__':
    main()
