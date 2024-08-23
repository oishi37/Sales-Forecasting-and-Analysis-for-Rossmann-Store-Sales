# Sales-Forecasting-and-Analysis-for-Rossmann-Store-Sales

Follow these steps to set up and run the project:

## 1. Clone the Repository

```bash
git clone https://github.com/oishi37/Sales-Forecasting-and-Analysis-for-Rossmann-Store-Sales.git
cd server
```

## 2. Download the csv files

From Kaggle https://www.kaggle.com/c/rossmann-store-sales/data download the datasets train.csv and store.csv and place both the files in the server folder.

## 3. Set Up the Virtual Environment

### For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Train the Model

```bash
python3 train.py
```

### Note: Close any windows of graphs and charts that pop up to continue.

## 6. Run the Server

```bash
python3 app.py
```

## 7. Run the Client

```bash
cd ..
cd client
```

### For macOS/Linux:

```bash
open index.html
```

### For Windows:

```bash
start index.html
```

## Introduction

This project focuses on analyzing and forecasting sales data for Rossmann stores using datasets provided by Kaggle. The objective is to build predictive models to forecast future sales and to gain insights into sales patterns across various stores. The analysis involves several key steps, including data preprocessing, feature engineering, the application of time series forecasting techniques and predictive model building. The project is implemented in a Jupyter Notebook and focuses on evaluating the performance of different regression models.

## Key Features

- **Time Series Analysis**: Identified cyclical patterns in the sales data using spectral analysis.
- **Machine Learning Models**:
  - **Gradient Boosting Regressor**: Gradient Boosting Regressor performed the best with a MAE of 686.10 and an R² of 90.27%, making it the most accurate model for this dataset.
  - **Random Forest Regressor**: Also performed well but slightly less accurate than Gradient Boosting.
  - **ElasticNet Regression**: Showed the least accuracy, indicating the limitations of linear models for this task.
- **Box-Cox Transformation**: Applied to stabilize variance and improve model performance.
- **Model Evaluation**: Metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score were used to assess model accuracy.

## Dataset

The data used here is from Kaggle https://www.kaggle.com/c/rossmann-store-sales/data. The dataset used for this project includes:

- **Sales Data**: Historical sales records.
- **Store Data**: Information about different stores, including types and locations.

## Models Used

- **Gradient Boosting Regressor**
- **Random Forest Regressor**
- **ElasticNet Regression**

## Feature Descriptions and Input Limitations

This project utilizes several features to predict sales. Below is a description of each feature and any input limitations:

Day of the Week (1-7): Numerical value representing the day of the week, where 1 = Monday and 7 = Sunday.
Number of Customers: The number of customers on a given day. This input is required for accurate predictions.
Promo (0 or 1): Indicates whether a store is running a promotion on that day (1 = Yes, 0 = No).
Assortment_a (0 or 1): Indicates whether the store has Assortment level 'a' (1 = Yes, 0 = No). Assortment 'a' represents a basic assortment.
Assortment_c (0 or 1): Indicates whether the store has Assortment level 'c' (1 = Yes, 0 = No). Assortment 'c' represents an extended assortment
Promo2 (0 or 1): Indicates whether the store is participating in Promo2, a continuous and consecutive promotion (1 = Yes, 0 = No).
Sales_Lag1: The sales value of the previous day. This is to capture the impact of sales trends from one day to the next.
Sales_Lag7: The sales value from the same day in the previous week. Useful for capturing weekly sales patterns.
Month (1-12): The month of the year, represented numerically where 1 = January and 12 = December.
Week of the Year (1-52): The week of the year, used to account for seasonality and other temporal effects.

## Input Limitations:

Numeric Inputs: Ensure that all numeric inputs are within the expected ranges (e.g., Days of the Week should be between 1 and 7).
Binary Inputs: Inputs like Promo, Assortment_a, Assortment_c, and Promo2 should only be 0 or 1.
Sales Lag Features: It is recommended to input realistic lag values based on historical sales data to maintain prediction accuracy. (Usually between 0 to 7)

## Conclusion

The project demonstrates the effectiveness of ensemble methods like Gradient Boosting and Random Forest for sales forecasting, particularly when dealing with non-linear and cyclical data. The Gradient Boosting Regressor is recommended for future forecasting tasks due to its superior accuracy.
