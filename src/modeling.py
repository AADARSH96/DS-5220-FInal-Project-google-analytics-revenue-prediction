import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Define models and their parameters for GridSearchCV
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {
            'classifier__alpha': [0.1, 1.0, 10.0]
        }
    },
    'Lasso Regression': {
        'model': Lasso(),
        'params': {
            'classifier__alpha': [0.1, 1.0, 10.0]
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
    'AdaBoost': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7, 10]
        }
    },
    'LightGBM': {
        'model': LGBMRegressor(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7, 10]
        }
    },
    'CatBoost': {
        'model': CatBoostRegressor(random_state=42, silent=True),
        'params': {
            'classifier__iterations': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__depth': [3, 5, 7, 10]
        }
    }
}

GridParams_LGBM = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
    'n_estimators': [30, 40, 60, 100, 200, 350, 400],
    'num_leaves': [6, 8, 12, 15, 16, 20, 25, 50, 100, 120, 150],
    'boosting_type': ['gbdt'],
    'objective': ['regression'],
    'metric': ['rmse'],
    'colsample_bytree': [0.6, 0.8, 1],
    'subsample': [0.7, 0.8, 0.9, 1],
    'reg_alpha': [0, 0.5, 0.8, 1],
    'reg_lambda': [0, 0.5, 0.8, 1],
    'min_child_samples': [1, 10, 20, 30]
}

GridParams_XGB = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
    'n_estimators': [30, 40, 60, 100, 200, 350, 400],
    'objective': ['reg:squarederror'],
    'colsample_bytree': [0.6, 0.8, 1],
    'subsample': [0.7, 0.8, 0.9, 1],
    'reg_alpha': [0, 0.5, 0.8, 1],
    'reg_lambda': [0, 0.5, 0.8, 1]
}

GridParams_GBDT = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
    'n_estimators': [30, 40, 60, 100, 200, 350, 400],
    'subsample': [0.7, 0.8, 0.9, 1],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [3, 5, 8, 10]
}

GridParams_RF = {
    'n_estimators': [60, 100, 200, 350, 400, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3]
}

models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {
            'classifier__alpha': [0.1, 1.0, 10.0]
        }
    },
    'Lasso Regression': {
        'model': Lasso(),
        'params': {
            'classifier__alpha': [0.1, 1.0, 10.0]
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
        'model': RandomizedSearchCV(RandomForestRegressor(random_state=42), GridParams_RF, scoring='neg_root_mean_squared_error', return_train_score=True, random_state=0, n_jobs=-1),
        'params': {}
    },
    'Gradient Boosting': {
        'model': RandomizedSearchCV(GradientBoostingRegressor(random_state=42), GridParams_GBDT, scoring='neg_root_mean_squared_error', return_train_score=True, random_state=0, n_jobs=-1),
        'params': {}
    },
    'Bagging': {
        'model': BaggingRegressor(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200]
        }
    },
    'AdaBoost': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    },
    'XGBoost': {
        'model': RandomizedSearchCV(XGBRegressor(random_state=42), GridParams_XGB, scoring='neg_root_mean_squared_error', return_train_score=True, random_state=0, n_jobs=-1),
        'params': {}
    },
    'LightGBM': {
        'model': RandomizedSearchCV(LGBMRegressor(random_state=42), GridParams_LGBM, scoring='neg_root_mean_squared_error', return_train_score=True, random_state=0, n_jobs=-1),
        'params': {}
    },
    'CatBoost': {
        'model': CatBoostRegressor(random_state=42, silent=True),
        'params': {
            'classifier__iterations': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__depth': [3, 5, 7, 10]
        }
    }
}

def evaluate_models(X_train, X_val, y_train, y_val):
    best_model = None
    best_r2 = -np.inf

    results = []
    for model_name, model_dict in models.items():
        print(f"Training {model_name}...")
        model = model_dict['model']
        params = model_dict['params']

        if model_name in ['XGBoost','Gradient Boosting','Random Forest','LightGBM']:
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)

        else:
            model_pipeline = Pipeline(steps=[
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
        print(model_results)
        results.append(model_results)

        if r2 > best_r2:
            best_r2 = r2
            best_model = best_estimator

        print(f"Finished training {model_name}.")

    print("All models have been trained.")
    combined_results_df = pd.DataFrame(results)

    return combined_results_df
