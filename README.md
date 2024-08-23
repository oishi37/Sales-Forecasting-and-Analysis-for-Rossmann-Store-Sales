# Sales-Forecasting-and-Analysis-for-Rossmann-Store-Sales
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


## Conclusion
The project demonstrates the effectiveness of ensemble methods like Gradient Boosting and Random Forest for sales forecasting, particularly when dealing with non-linear and cyclical data. The Gradient Boosting Regressor is recommended for future forecasting tasks due to its superior accuracy.
