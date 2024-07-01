# DS-5220-FInal-Project-google-analytics-revenue-prediction

This project aims to predict revenue per customer using the Google Merchandise Store (GStore) dataset. The project involves data loading, preprocessing, feature engineering, model training, and evaluation. The goal is to derive actionable insights and make better use of marketing budgets through data analysis.

PROJECT FLOW
![Untitled-2024-01-18-1940](https://github.com/AADARSH96/DS-5220-FInal-Project-google-analytics-revenue-prediction/assets/30507087/11b13394-ed6f-4fb8-967c-1fef2a8e0694)

PROJECT STRUCTURE:<br/> 
project_name/ <br/> 
│<br/> 
├── data/<br/> 
│   ├── train.csv<br/> 
│   └── test.csv<br/> 
│<br/> 
├── notebooks/<br/> 
│   ├──  iteration3.ipynb<br/> 
│<br/> 
├── src/<br/> 
│   ├── __init__.py<br/> 
│   ├── data_loader.py<br/> 
│   ├── data_preprocessing.py<br/> 
│   ├── visualization.py<br/> 
│   ├── modeling.py<br/> 
│   ├── constants.py<br/> 
│<br/> 
├── main.py<br/> 
├── requirements.txt<br/> 
└── README.md<br/> 

data/: Contains the raw CSV files for training and testing.<br/> 
notebooks/: Contains Jupyter notebooks for exploratory data analysis and model evaluation.

src/: Contains the source code, organized by functionality.<br/> 
- __init__.py: Initializes the src package.
- data_loader.py: Functions for loading and processing raw data.
- data_preprocessing.py: Functions for data preprocessing and feature engineering.
- visualization.py: Functions for data visualization.
- modeling.py: Functions for model training and evaluation.
- constants.py: Constants used throughout the project.

main.py: Main script to run the entire pipeline.<br/> 
requirements.txt: Python package dependencies.<br/> 
README.md: Project documentation.<br/> 


SETUP:<br/>
git clone https://github.com/yourusername/project_name.git<br/>
cd project_name<br/>
pip install -r requirements.txt<br/>


Data Loading<br/>
The data loading module (data_loader.py) reads the raw CSV files, processes JSON columns, and handles missing values. It prints the shape of the data and the columns dropped due to having a single unique value or a high percentage of missing values.

Data Preprocessing<br/>
The preprocessing module (data_preprocessing.py) handles data cleaning and transformation. It includes functions to preprocess data and transform the target variable (totals.transactionRevenue) into its logarithmic form for better model performance.

Visualization<br/>
The visualization module (visualization.py) provides functions to plot various distributions in the data, such as channel grouping, browser usage, operating systems, and mobile vs. non-mobile visits.

Modeling<br/>
The modeling module (modeling.py) defines the machine learning pipeline, including data preprocessing, model training, and evaluation. It supports various regression models, such as Linear Regression, Ridge Regression, and Lasso Regression. Model performance is evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.

Results<br/>
Model evaluation results are printed in a tabular format using the tabulate library, showcasing the performance of each model.


