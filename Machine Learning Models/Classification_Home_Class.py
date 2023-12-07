#%%
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.feature_selection.significance_tests import target_binary_feature_real_test
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


# Load your dataset
file_path = r"C:\Users\Windows\OneDrive\Documents\Research\New folder\Dataset1\Enhanced_Processed_IAQ_and_Survey_Cleaned.xlsx"
data = pd.read_excel(file_path)
Features_path = r"C:\Users\Windows\OneDrive\Documents\Research\New folder\Dataset1\New_Features.xlsx"
features = pd.read_excel(Features_path)



# Function to convert price to home class
def price_to_class(price):
    if price < 1000000:
        return 1
    elif 1000000 <= price < 1500000:
        return 2
    else:
        return 3

# Convert Total_Price to Home_Class
features['Home_Class'] = features['Total_Price'].apply(price_to_class)

# Define your features and target
X = features.drop(['Home_Class', 'Home_ID', 'Total_Price'], axis=1)  # Exclude non-feature columns
y = features['Home_Class']

# Apply SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate classifier performance
y_pred = rf.predict(X_test)
# print(classification_report(y_test, y_pred))

# Feature importance evaluation
importances = rf.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plotting feature importances
# plt.figure(figsize=(30, 16))
# feature_importances.plot(kind='bar')
# plt.title('Feature Importances in RandomForest Classifier')
# plt.ylabel('Importance')
# plt.xlabel('Features')
# plt.show()

##GridSearch
# Hyperparameters grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
# print("Best Parameters:", grid_search.best_params_)

# Train RandomForest with best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Evaluate the optimized model
y_pred_optimized = best_rf.predict(X_test)
# print("\nOptimized Model Performance:")
# print(classification_report(y_test, y_pred_optimized))

# Feature importance evaluation for the optimized model
optimized_importances = best_rf.feature_importances_
optimized_feature_importances = pd.Series(optimized_importances, index=feature_names).sort_values(ascending=False)

# Plotting optimized feature importances
# plt.figure(figsize=(10, 8))
# optimized_feature_importances.plot(kind='bar')
# plt.title('Optimized Feature Importances in RandomForest Classifier')
# plt.ylabel('Importance')
# plt.xlabel('Features')
# plt.show()
# print(optimized_feature_importances[optimized_feature_importances > 0.025])

# Select features with importance greater than 0.025
selected_features = optimized_feature_importances[optimized_feature_importances > 0.025].index

# Extract these features from the dataset
X_selected = X[selected_features]

# Split the data into training and testing sets
X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomForestClassifier
rf_selected = RandomForestClassifier(random_state=42)

# Grid search
grid_search_selected = GridSearchCV(estimator=rf_selected, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_selected.fit(X_train_selected, y_train_selected)

# Best parameters
print("Best Parameters:", grid_search_selected.best_params_)

# Train RandomForest with best parameters
best_rf_selected = grid_search_selected.best_estimator_
best_rf_selected.fit(X_train_selected, y_train_selected)

# Evaluate the optimized model
y_pred_selected = best_rf_selected.predict(X_test_selected)
print("\nOptimized Model Performance (Selected Features):")
print(classification_report(y_test_selected, y_pred_selected))
