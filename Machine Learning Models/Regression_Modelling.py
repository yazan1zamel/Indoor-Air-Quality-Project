
#%%
# Cell 1. Reading required packages 
# Cell2 : Reading class 
# Cell 3: PM2.5 Regression Modelling Process 
# Cell 4: CO2 Regression Modelling Process 
# Cell 5: NO2 Regression Modelling Process 
# Cell 6: Temperature(T) Regression Modelling Process 
# Cell 7: Relative Humidity (RH) Regression Modelling Process 
## Each cell includes all the codes we used to visualize, normalize, implement PCA, grid search,.... 

#%%
# Importing Required Packages

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.feature_selection.significance_tests import target_binary_feature_real_test
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression



#%%
# Reading dataset including all features
Features_path = r"C:\Users\Windows\OneDrive\Documents\Research\New folder\Dataset1\New_Features.xlsx"
features = pd.read_excel(Features_path)

#%%
# ------------------------------------------------------------------------------------------------------

# PM2.5 Regression Model 

# Dropping non-feature columns (if any)
X = features.drop(['Home_ID', 'Home_Class'], axis=1)  

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Applying PCA
# pca = PCA()
# pca.fit(X_scaled)

# # Plotting the Cumulative Summation of the Explained Variance (Elbow Graph)
# plt.figure(figsize=(20, 12))
# plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
# plt.title('Explained Variance by Components')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.show()

# Applying PCA
# pca = PCA(n_components=5)
# pca.fit(X_scaled)

# X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, features['PM_MET_IN1'], test_size=0.2, random_state=42)


# Initialize and train the regression model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predicting on the test set
y_pred = regressor.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)



print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")

importances = regressor.feature_importances_
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# print("Feature Importances:")
print(feature_importances)


#Grid Search 
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the grid search model
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters
# print("Best Parameters:", grid_search.best_params_)

# Best model from grid search
best_regressor = grid_search.best_estimator_

# Perform cross-validation and calculate the mean score
cv_scores = cross_val_score(best_regressor, X, features['PM_MET_IN1'], cv=5)
# print("Cross-validation scores:", cv_scores)
# print("Mean cross-validation score:", cv_scores.mean())

# Correlation Analysis
correlation, _ = pearsonr(features['Total_Price'], features['PM_MET_IN1'])
# print("Pearson Correlation Coefficient:", correlation)

# Simple Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(features[['Total_Price']], features['PM_MET_IN1'])

# Plotting the regression line
plt.scatter(features['Total_Price'], features['PM_MET_IN1'], alpha=0.5)
plt.plot(features['Total_Price'], lin_reg.predict(features[['Total_Price']]), color='red')
plt.xlabel('Total Price')
plt.ylabel('PM_MET_IN1')
plt.title('Linear Regression: Total Price vs PM_MET_IN1')
plt.show()


# Assuming 'best_regressor' is the best model obtained from GridSearchCV
y_pred = best_regressor.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Plotting residuals
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
# ---------------------------------------------------------------------------------------------------------------------------------
#%%
# CO2 Regression Model 
X = features.drop(['CO2_ETC_IN1','Home_ID', 'Home_Class'], axis=1)  # Drop the target variable from features
y = features['CO2_ETC_IN1']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predicting on the test set
y_pred = regressor.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")

importances = regressor.feature_importances_
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# print("Feature Importances:")
print(feature_importances)


# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_


# Evaluate the best model from GridSearchCV using cross-validation
best_regressor = grid_search.best_estimator_
cv_scores = cross_val_score(best_regressor, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

-------------------------------------------------------------
#Checking teh direct correlation between teh home price and the CO2 levels

# Define the variables
X = features[['Total_Price']]  # Predictor
y = features['CO2_ETC_IN1']    # Response

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Price', y='CO2_ETC_IN1', data=features, label='Actual Data')
plt.plot(features['Total_Price'], y_pred, color='red', label='Regression Line')
plt.xlabel('Total Price')
plt.ylabel('CO2 ETC IN1')
plt.title('Regression Analysis: Total Price vs CO2 ETC IN1')
plt.legend()
plt.show()
correlation = features[['Total_Price', 'CO2_ETC_IN1']].corr().iloc[0, 1]
print(f"Pearson Correlation Coefficient: {correlation}")

----------------------------------
# Predicting on the test set

y_pred = best_regressor.predict(X_test)

# Residuals
residuals = y_test - y_pred

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Observed Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

sns.set_context("paper", font_scale=0.7)

correlation_matrix = features.corr()

# Focusing on CO2_ETC_IN1 correlations
co2_correlations = correlation_matrix['CO2_ETC_IN1'].sort_values(ascending=False)
print(co2_correlations)
# # Plotting
# Plotting
plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap with Respect to CO2 Levels")
plt.show()

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying PCA
pca = PCA(n_components=40)  # Retaining all features
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

regressor = RandomForestRegressor(n_estimators=40, max_depth= 15, min_samples_split= 5,  random_state=42)
regressor.fit(X_train_pca, y_train)

# Predicting on the test set
y_pred = regressor.predict(X_test_pca)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")


# Assuming 'features' is your DataFrame
X = features[['Year_Built', 'Home_Size_sq_ft', 'Number_of_Occupants', 
              'heating_appliance_Gas_fireplace_log_set', 'Weekly_Shower_Usage', 
              'Weekly_Bath_Jacuzzi_Usage', 'Weekly_Dishwasher_Usage', 
              'Weekly_Washing_Machine_Usage', 'Weekly_Indoor_Clothes_Drying', 
              'Frequency_Indoor_Smoking', 'Frequency_Using_Candles_Incense', 
              'Frequency_Vacuuming', 'Frequency_Using_Floor_Cleaning_Agents', 
              'Frequency_Using_Air_Fresheners', 'Frequency_Using_Pesticide_Spray', 
              'Frequency_Using_Paints_Glue_Solvents', 'Frequency_Using_Humidifier', 
              'Frequency_Using_Dehumidifier', 'Use_Air_Fresheners', 
              'Shoes_Worn_Indoors', 'Number_of_Pets', 'Use_Portable_Air_Filter', 
              'PM_MET_IN1', 'NO2_AQL_IN1', 'T_ETC_IN1', 'RH_ETC_IN1', 
              'Total_Price', 'Avg_Cooking_Frequency', 'Avg_Window_Open_Hours']]

y = features['CO2_ETC_IN1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 50,30,20,10,5,31,32,33,34,35,38,39,40,12,13,14,15,17,18,19],
    'learning_rate': [0.01, 0.05, 0.1, 0.2,0.3,0.4,0.5, 0.005],
    'max_depth': [3, 4, 5,6,10,15]
}

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Best estimator
best_gbr = grid_search.best_estimator_

# Predicting on the test set
y_pred = best_gbr.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

Display results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")


# Access the feature importances
importances = best_gbr.feature_importances_

# Create a series of feature importances
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Print feature importances
print(feature_importances)

# Selecting features and target
X = features[['Home_Size_sq_ft', 
              'NO2_AQL_IN1', 'T_ETC_IN1', 'RH_ETC_IN1', 'Avg_Cooking_Frequency', 'Avg_Window_Open_Hours','Frequency_Using_Humidifier ']]
y = features['CO2_ETC_IN1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalizing the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 50, 30, 20, 10, 5, 31, 32, 33, 34, 35, 38, 39, 40, 12, 13, 14, 15, 17, 18, 19],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.005],
    'max_depth': [3, 4, 5, 6, 10, 15]
}

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

# Fit the GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Best estimator
best_gbr = grid_search.best_estimator_

# Predicting on the test set
y_pred = best_gbr.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")


# Access the feature importances
importances = best_gbr.feature_importances_

# Create a series of feature importances
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Print feature importances
print(feature_importances)

#%%
# NO2 Regression Model 


# Selecting features and target
X = features[[ 'Weekly_Dishwasher_Usage', 
               'Weekly_Washing_Machine_Usage',  
                'Number_of_Pets',  
               'PM_MET_IN1', 'T_ETC_IN1', 'RH_ETC_IN1', 
               'Avg_Cooking_Frequency', 'Avg_Window_Open_Hours']]

y = features['NO2_AQL_IN1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalizing the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 50, 30, 20, 10, 5, 31, 32, 33, 34, 35, 38, 39, 40, 12, 13, 14, 15, 17, 18, 19],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.005],
    'max_depth': [3, 4, 5, 6, 10, 15]
}

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

# Fit the GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Best estimator
best_gbr = grid_search.best_estimator_

# Predicting on the test set
y_pred = best_gbr.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")



# Access the feature importances
importances = best_gbr.feature_importances_

# Create a series of feature importances
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Print feature importances
print(feature_importances)


# --------------------------
# Checking teh direct correlation between teh home price and the CO2 levels

# Define the variables
X = features[['Total_Price']]  # Predictor
y = features['NO2_AQL_IN1']    # Response

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Price', y='NO2_AQL_IN1', data=features, label='Actual Data')
plt.plot(features['Total_Price'], y_pred, color='red', label='Regression Line')
plt.xlabel('Total Price')
plt.ylabel('CO2 ETC IN1')
plt.title('Regression Analysis: Total Price vs NO2 ETC IN1')
plt.legend()
plt.show()
correlation = features[['Total_Price', 'NO2_AQL_IN1']].corr().iloc[0, 1]
print(f"Pearson Correlation Coefficient: {correlation}")

#%%
# Temperature (T) Regression Model 

# Selecting features and target
X = features[['Home_Size_sq_ft', 
             'NO2_AQL_IN1', 'CO2_ETC_IN1','RH_ETC_IN1', 
              'Total_Price', 'Avg_Cooking_Frequency', 'Avg_Window_Open_Hours']]

y = features['T_ETC_IN1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizing the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 50, 30, 20, 10, 5, 31, 32, 33, 34, 35, 38, 39, 40, 12, 13, 14, 15, 17, 18, 19],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.005],
    'max_depth': [3, 4, 5, 6, 10, 15]
}

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

# Fit the GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Best estimator
best_gbr = grid_search.best_estimator_

# Predicting on the test set
y_pred = best_gbr.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")



# Access the feature importances
importances = best_gbr.feature_importances_

# Create a series of feature importances
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Print feature importances
print(feature_importances)


----------------
# Checking teh direct correlation between teh home price and the CO2 levels

# Define the variables
X = features[['Total_Price']]  # Predictor
y = features['T_ETC_IN1']    # Response

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Price', y= 'T_ETC_IN1', data=features, label='Actual Data')
plt.plot(features['Total_Price'], y_pred, color='red', label='Regression Line')
plt.xlabel('Total Price')
plt.ylabel( 'T_ETC_IN1')
plt.title('Regression Analysis: Total Price vs Temperature ETC IN1')
plt.legend()
plt.show()
correlation = features[['Total_Price', 'T_ETC_IN1']].corr().iloc[0, 1]
print(f"Pearson Correlation Coefficient: {correlation}")
#%%
# Relative Humidity (RH) Regression Model 

# Selecting features and target
X = features[[ 'Home_Size_sq_ft', 'Number_of_Occupants', 
            'Weekly_Shower_Usage', 'Frequency_Using_Candles_Incense', 
              'Frequency_Using_Paints_Glue_Solvents', 'Frequency_Using_Humidifier',          
              'PM_MET_IN1', 'NO2_AQL_IN1', 'T_ETC_IN1', 'CO2_ETC_IN1',
               'Avg_Cooking_Frequency', 'Avg_Window_Open_Hours','Number_of_Pets']]
# X = features[[ 'Home_Size_sq_ft',      
#               'PM_MET_IN1', 'NO2_AQL_IN1', 'T_ETC_IN1', 'CO2_ETC_IN1']]
y = features['RH_ETC_IN1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 50, 30, 20, 10, 5, 31, 32, 33, 34, 35, 38, 39, 40, 12, 13, 14, 15, 17, 18, 19],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.005,0.07],
    'max_depth': [3, 4, 5, 6, 10, 15,2]
}

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

# Fit the GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Best estimator
best_gbr = grid_search.best_estimator_

# Predicting on the test set
y_pred = best_gbr.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")



# Access the feature importances
importances = best_gbr.feature_importances_

# Create a series of feature importances
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Print feature importances
print(feature_importances)


# -------------------------------------------
# Checking teh direct correlation between teh home price and the CO2 levels

# Define the variables
X = features[['Total_Price']]  # Predictor
y = features[ 'RH_ETC_IN1']    # Response

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Price', y='RH_ETC_IN1', data=features, label='Actual Data')
plt.plot(features['Total_Price'], y_pred, color='red', label='Regression Line')
plt.xlabel('Total Price')
plt.ylabel('RH_ETC_IN1')
plt.title('Regression Analysis: Total Price vs Relative Humidity')
plt.legend()
plt.show()
# correlation = features[['Total_Price', 'RH_ETC_IN1']].corr().iloc[0, 1]
# print(f"Pearson Correlation Coefficient: {correlation}")
