# DS-5220-FInal-Project-google-analytics-revenue-prediction

This project aims to predict revenue per customer using the Google Merchandise Store (GStore) dataset. The project involves data loading, preprocessing, feature engineering, model training, and evaluation. The goal is to derive actionable insights and make better use of marketing budgets through data analysis.

dataset - https://www.kaggle.com/competitions/ga-customer-revenue-prediction/data

<b>PROJECT FLOW</b>
<img width="698" alt="Screenshot 2024-07-19 at 8 38 03 PM" src="https://github.com/user-attachments/assets/e19b49ae-411b-4396-b414-68461aaab84b">
</br>
</br>



<b>PROJECT STRUCTURE:</b><br/> 
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

<b>data/:</b> Contains the raw CSV files for training and testing.<br/> 
<b>notebooks/:</b> Contains Jupyter notebooks for exploratory data analysis and model evaluation.

<b>src/:</b> Contains the source code, organized by functionality.<br/> 
- <b>__init__.py:</b> Initializes the src package.
- <b>data_loader.py:</b> Functions for loading and processing raw data.
- <b>data_preprocessing.py:</b> Functions for data preprocessing and feature engineering.
- <b>visualization.py:</b> Functions for data visualization.
- <b>modeling.py:</b> Functions for model training and evaluation.
- <b>constants.py:</b> Constants used throughout the project.

<b>main.py:</b> Main script to run the entire pipeline.<br/> 
<b>requirements.txt:</b> Python package dependencies.<br/> 
<b>README.md:</b> Project documentation.<br/> 


<b>SETUP:</b><br/>
git clone https://github.com/yourusername/project_name.git<br/>
cd project_name<br/>
pip install -r requirements.txt<br/>


<b>Data Loading</b><br/>
The data loading module (data_loader.py) reads the raw CSV files, processes JSON columns, and handles missing values. It prints the shape of the data and the columns dropped due to having a single unique value or a high percentage of missing values.

<b>Data Preprocessing</b><br/>
The preprocessing module (data_preprocessing.py) handles data cleaning and transformation. It includes functions to preprocess data and transform the target variable (totals.transactionRevenue) into its logarithmic form for better model performance.

<b>Visualization</b><br/>
The visualization module (visualization.py) provides functions to plot various distributions in the data, such as channel grouping, browser usage, operating systems, and mobile vs. non-mobile visits.

<b>Modeling</b><br/>
The modeling module (modeling.py) defines the machine learning pipeline, including data preprocessing, model training, and evaluation. It supports various regression models, such as Linear Regression, Ridge Regression, and Lasso Regression. Model performance is evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.

<b>Results</b><br/>
Model evaluation results are printed in a tabular format using the tabulate library, showcasing the performance of each model.


