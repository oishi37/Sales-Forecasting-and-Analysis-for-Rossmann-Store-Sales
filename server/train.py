# Installing libraries
from statsmodels.tsa.stattools import kpss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from numpy.fft import fft
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
from scipy.stats import boxcox
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.stats import chi2
import pickle

# Load datasets with explicit dtypes to avoid DtypeWarning
print("Loading data...")
train_df = pd.read_csv('train.csv', dtype={
                       'StateHoliday': 'str', 'Sales': 'float64', 'Customers': 'int64', 'Open': 'int64'}, low_memory=False)
store_df = pd.read_csv('store.csv', dtype={
                       'CompetitionDistance': 'float64', 'Promo2SinceYear': 'float64'}, low_memory=False)

# Impute missing values using median or mode, with loc to avoid chained assignment warning
print("Preprocessing data...")
train_df['StateHoliday'] = train_df['StateHoliday'].astype(str)

store_df.loc[:, 'CompetitionDistance'] = store_df['CompetitionDistance'].fillna(
    store_df['CompetitionDistance'].median())
store_df.loc[:, 'CompetitionOpenSinceMonth'] = store_df['CompetitionOpenSinceMonth'].fillna(
    store_df['CompetitionOpenSinceMonth'].mode()[0])
store_df.loc[:, 'CompetitionOpenSinceYear'] = store_df['CompetitionOpenSinceYear'].fillna(
    store_df['CompetitionOpenSinceYear'].mode()[0])
store_df.loc[:, 'Promo2SinceWeek'] = store_df['Promo2SinceWeek'].fillna(
    store_df['Promo2SinceWeek'].mode()[0])
store_df.loc[:, 'Promo2SinceYear'] = store_df['Promo2SinceYear'].fillna(
    store_df['Promo2SinceYear'].mode()[0])

# convert Date columns to datetime
train_df['Date'] = pd.to_datetime(train_df['Date'])


# check the datasets again
# print(train_df.info())
# print(test_df.info())
# print(store_df.info())


# List of months to create columns for
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

# Initialize new columns for each month with 0
for month in months:
    store_df[month] = 0

# Populate the new month columns based on PromoInterval
for i, row in store_df.iterrows():
    if pd.notnull(row['PromoInterval']):
        promo_months = row['PromoInterval'].split(',')
        for month in promo_months:
            store_df.at[i, month] = 1

# Drop the original PromoInterval column if no longer needed
store_df.drop('PromoInterval', axis=1, inplace=True)

# Merging datasets
merged_df = pd.merge(train_df, store_df, how='left', on='Store')


# Droping Open = 0, since if store is closed, makes no sense to see the sales
merged_df = merged_df[merged_df['Open'] == 1]

# print(store_df.info())
###############


###############
# Extract Date Features like Year, Month, Day, WeekOfYear to help in capturing seasonality and trends.
print("Extracting features...")
merged_df['Year'] = merged_df['Date'].dt.year
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Day'] = merged_df['Date'].dt.day
merged_df['WeekOfYear'] = merged_df['Date'].dt.isocalendar().week

# Lag features for time series forecasting.

merged_df['Sales_Lag1'] = merged_df.groupby('Store')['Sales'].shift(1)
merged_df['Sales_Lag7'] = merged_df.groupby('Store')['Sales'].shift(7)


# Encode Categorical Variables
# One-Hot Encoding for StoreType and Assortment
merged_df = pd.get_dummies(merged_df, columns=['Assortment'])
merged_df = pd.get_dummies(merged_df, columns=['StateHoliday'])


# Scaling ensures that all features contribute equally to the analysis: normalization and standardization
print("Normalizing features...")
# Initialize the scaler
scaler = StandardScaler()

# List of columns to scale
columns_to_scale = ['Sales_Lag1', 'Sales_Lag7',
                    'CompetitionDistance', 'Customers']

# Apply scaling
merged_df[columns_to_scale] = scaler.fit_transform(merged_df[columns_to_scale])

# STEP 3
# time series decomposition
daily_sales = merged_df.groupby('Date')['Sales'].sum().reset_index()

decomposition = seasonal_decompose(
    # period can be adjusted
    daily_sales['Sales'], model='multiplicative', period=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition.plot()
plt.show()

# trend detection
daily_sales['Sales_MA'] = daily_sales['Sales'].rolling(
    window=30).mean()  # 30-day moving average
daily_sales[['Date', 'Sales', 'Sales_MA']].plot(x='Date', figsize=(14, 7))
plt.show()


# Prepare the data
daily_sales['Date_ordinal'] = pd.to_datetime(
    daily_sales['Date']).map(lambda date: date.toordinal())
X = daily_sales['Date_ordinal'].values.reshape(-1, 1)
y = daily_sales['Sales'].values

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

# the original data and the trend line
plt.figure(figsize=(10, 6))
plt.plot(daily_sales['Date'], daily_sales['Sales'], label='Original Sales')
plt.plot(daily_sales['Date'], trend, color='red',
         label='Linear Trend', linewidth=2)
plt.legend()
plt.title('Sales with Linear Trend')
plt.show()

# Seasonlity detection
monthly_sales = merged_df.groupby(merged_df['Date'].dt.month)['Sales'].sum()
monthly_sales.plot(kind='bar', figsize=(10, 5))
plt.show()


# Fourier transform to detect cycles within the data.
sales_fft = fft(daily_sales['Sales'])
plt.plot(np.abs(sales_fft))
plt.show()


# SARIMA to Predict future sales considering both trend and seasonality.

# test for stationary
# ADF Test
adf_result = adfuller(daily_sales['Sales'])
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])

# KPSS Test
kpss_result = kpss(daily_sales['Sales'], regression='c')
print('KPSS Statistic:', kpss_result[0])
print('p-value:', kpss_result[1])

# ACF and PACF plots
fig, ax = plt.subplots(2, figsize=(12, 8))

plot_acf(daily_sales['Sales'].dropna(), ax=ax[0])
plot_pacf(daily_sales['Sales'].dropna(), ax=ax[1])

plt.show()

# the SARIMA model
model = SARIMAX(daily_sales['Sales'],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

# Fit the model

results = model.fit()
print(results.summary())
daily_sales['forecast'] = results.predict(
    start=0, end=len(daily_sales)-1, dynamic=False)
daily_sales[['Sales', 'forecast']].plot(figsize=(10, 5))
plt.show()

# check the residue since forcast closelsy follows the data, so like overfitting
# Extracting the residuals from the fitted model
residuals = results.resid

# Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.show()

# QQ-Plot
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()

# Box-Cox transformation
daily_sales['BoxCox_Sales'], fitted_lambda = boxcox(
    daily_sales['Sales'] + 1)  # Adding 1 to handle zeros

# Fit SARIMA model on Box-Cox transformed data
model = SARIMAX(daily_sales['BoxCox_Sales'],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()

# the residuals
residuals = results.resid

# histogram of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals (After Box-Cox Transformation)')
plt.show()

# QQ-Plot
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals (After Box-Cox Transformation)')
plt.show()

#  the fitted lambda
print("Fitted Lambda for Box-Cox Transformation:", fitted_lambda)

# normality is better but not fixed.

# After the box-cox spectral Analysis
# Extract residuals from SARIMA model
residuals = results.resid

# Compute the periodogram (spectral density)
frequencies, spectral_density = periodogram(residuals, scaling='spectrum')

# the predominant frequency
predominant_freq_index = np.argmax(spectral_density)
predominant_frequency = frequencies[predominant_freq_index]
max_spectral_density = spectral_density[predominant_freq_index]

# confidence intervals for the spectral density
alpha = 0.05  # 95% confidence level
dof = 2  # Degrees of freedom for the periodogram

# confidence interval
lower_bound = max_spectral_density * chi2.ppf(alpha / 2, dof) / dof
upper_bound = max_spectral_density * chi2.ppf(1 - alpha / 2, dof) / dof

# the results
print(f"Predominant Frequency: {predominant_frequency}")
print(f"Spectral Density at Predominant Frequency: {max_spectral_density}")
print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")

# the periodogram with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(frequencies, spectral_density, label='Spectral Density')
plt.axvline(predominant_frequency, color='r',
            linestyle='--', label='Predominant Frequency')
plt.fill_between(frequencies, lower_bound, upper_bound,
                 color='gray', alpha=0.3, label='95% Confidence Interval')
plt.xlabel('Frequency')
plt.ylabel('Spectral Density')
plt.title('Periodogram of Residuals')
plt.legend()
plt.show()


####
# Label encode the StoreType column
label_encoder = LabelEncoder()
merged_df['StoreType'] = label_encoder.fit_transform(merged_df['StoreType'])

# Proceed with your analysis or modeling
# Example: Aggregating total sales by store
store_sales = merged_df.groupby(['Store', 'StoreType'])[
    'Sales'].sum().reset_index()

# Aggregate average sales by store type
store_type_sales = store_sales.groupby(
    'StoreType')['Sales'].mean().reset_index()

# Bar Plot for Average Sales by Store Type:
plt.figure(figsize=(10, 6))
sns.barplot(x='StoreType', y='Sales', data=store_type_sales)
plt.title('Average Sales by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Average Sales')
plt.show()

# Box Plot to Show Distribution of Sales within Each Store Type:
plt.figure(figsize=(12, 8))
sns.boxplot(x='StoreType', y='Sales', data=store_sales)
plt.title('Sales Distribution by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Total Sales per Store')
plt.show()
####


####
# Top 5 Stores by Sales from Each Store Type
top_stores = store_sales.groupby('StoreType').apply(
    lambda x: x.nlargest(5, 'Sales')).reset_index(drop=True)

plt.figure(figsize=(14, 8))
sns.barplot(x='Store', y='Sales', hue='StoreType', data=top_stores)
plt.title('Top 5 Stores by Sales for Each Store Type')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# Heatmap for Sales Performance Across a Selection of Stores:
# a sample of 50 stores to avoid overcrowding
sampled_stores = store_sales.sample(50, random_state=42)

# data for the heatmap
sales_pivot = sampled_stores.pivot(
    index="Store", columns="StoreType", values="Sales")

# heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(sales_pivot, annot=False, cmap="YlGnBu")
plt.title('Sales Heatmap for a Sample of 50 Stores')
plt.xlabel('Store Type')
plt.ylabel('Store')
plt.show()
####

merged_df = merged_df.fillna(0)

# List of columns to include in X
feature_columns = [
    'Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday_a',
    'StateHoliday_b', 'StateHoliday_c', 'SchoolHoliday', 'StoreType', 'Assortment_a',
    'Assortment_b', 'Assortment_c', 'CompetitionDistance', 'Promo2',
    'Promo2SinceWeek', 'Promo2SinceYear', 'Sales_Lag1', 'Sales_Lag7',
    'Month', 'Year', 'WeekOfYear'
]

# the feature matrix X
X = merged_df[feature_columns]

y = merged_df['Sales']


# Apply SelectKBest to extract the top 10 features
bestfeatures = SelectKBest(score_func=f_regression, k=10)
X_reduced = bestfeatures.fit_transform(X, y)

# mask of selected features
selected_features_mask = bestfeatures.get_support()

selected_features = np.array(X.columns)[selected_features_mask]

print("Selected features are:", selected_features)

X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []
# Looping through each store
for store_id in merged_df['Store'].unique():
    store_data = merged_df[merged_df['Store'] == store_id]

    # putting the index for splitting based on time for this store
    split_index = int(len(store_data) * 0.8)

    # then split the data for this store
    X_train_list.append(store_data.iloc[:split_index].drop('Sales', axis=1))
    X_test_list.append(store_data.iloc[split_index:].drop('Sales', axis=1))
    y_train_list.append(store_data.iloc[:split_index]['Sales'])
    y_test_list.append(store_data.iloc[split_index:]['Sales'])


X_train = pd.concat(X_train_list)
X_test = pd.concat(X_test_list)
y_train = pd.concat(y_train_list)
y_test = pd.concat(y_test_list)

# Droping 'Store' column from X_train and X_test
X_train = X_train.drop(columns=['Store'])
X_test = X_test.drop(columns=['Store'])

# Ensure X_train and X_test have the same columns (selected features)
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# missing values in lag features in X_train
X_train['Sales_Lag1'].fillna(0, inplace=True)
X_train['Sales_Lag7'].fillna(0, inplace=True)

print("Training Random Forest Regressor...")
# Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# metrics for Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf) * 100  # R² score as a percentage


# Convert boolean columns to integers
X_train['Assortment_a'] = X_train['Assortment_a'].astype(int)
X_train['Assortment_c'] = X_train['Assortment_c'].astype(int)
X_test['Assortment_a'] = X_test['Assortment_a'].astype(int)
X_test['Assortment_c'] = X_test['Assortment_c'].astype(int)


# Gradient Boosting Regressor
print("Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Predictions on the test set
y_pred_gb = gb_model.predict(X_test)

# metrics for Gradient Boosting
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb) * 100  # R² score as a percentage

# Elastic Net Regressor
print("Training Elastic Net Regressor...")
en_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
en_model.fit(X_train, y_train)

# Predictions on the test set
y_pred_en = en_model.predict(X_test)

# Calculate metrics for Elastic Net
mae_en = mean_absolute_error(y_test, y_pred_en)
mse_en = mean_squared_error(y_test, y_pred_en)
rmse_en = np.sqrt(mse_en)
r2_en = r2_score(y_test, y_pred_en) * 100  # R² score as a percentage

# Re-train the Gradient Boosting Regressor on the entire dataset
gbr_model = GradientBoostingRegressor(
    n_estimators=100, max_depth=5, random_state=42)
gbr_model.fit(X_reduced, y)

future_sales_predictions = gbr_model.predict(X_reduced)

# Plotting the predicted future sales
plt.figure(figsize=(10, 6))
plt.plot(future_sales_predictions, label='Predicted Sales')
plt.title('Future Sales Predictions')
plt.xlabel('Time (e.g., days, weeks)')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Save the model
print("Saving trained model...")
with open('model.pkl', 'wb') as file:
    pickle.dump(gbr_model, file)

print("Saved.")
