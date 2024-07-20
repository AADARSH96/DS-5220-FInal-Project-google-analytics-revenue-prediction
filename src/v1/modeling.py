from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from tabulate import tabulate


def define_preprocessor(numerical_features, categorical_features):
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    return preprocessor


def train_and_evaluate_models(X, y, preprocessor):
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Ridge Regression': {
            'model': Ridge(),
            'params': {
                'classifier__alpha': [0.1, 1.0, 10.0]  # Example values for alpha
            }
        },
        'Lasso Regression': {
            'model': Lasso(),
            'params': {
                'classifier__alpha': [0.1, 1.0, 10.0]  # Example values for alpha
            }
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7, 10]
            }
        },
        'Bagging': {
            'model': BaggingRegressor(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200]
            }
        },
        'Support Vector Machine': {
            'model': SVR(),
            'params': {
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__C': [0.1, 1, 10]
            }
        }
    }

    combined_results = []

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, model_dict in models.items():
        model = model_dict['model']
        params = model_dict['params']

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        grid_search = GridSearchCV(model_pipeline, param_grid=params, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        val_preds = best_estimator.predict(X_val)

        mae = mean_absolute_error(y_val, val_preds)
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, val_preds)

        model_results = {
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2 Score': r2,
        }

        combined_results.append(model_results)

    return pd.DataFrame(combined_results)


def print_results(results_df):
    print(tabulate(results_df, headers='keys', tablefmt='psql'))
