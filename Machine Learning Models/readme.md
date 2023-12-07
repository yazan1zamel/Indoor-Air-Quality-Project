Comprehensive Analysis of Indoor Air Quality Prameters and Home Classification
Overview

This repository encompasses a collection of Python scripts for an in-depth analysis of indoor air quality and its influence on home classification and valuation. The project integrates data cleaning, feature engineering, regression and classification modeling, and data merging techniques, offering a robust analysis of how indoor environmental factors affect home classification based on price.

Data Cleaning and Preprocessing

- Standardization: Vital features, especially those related to indoor air quality and satisfaction, are standardized for data consistency.
Data Transformation: Implement transformations like mapping categorical data to numerical values, replacing missing values, and dropping irrelevant columns.
- Principal Component Analysis (PCA): Used to reduce the dimensionality of satisfaction-related features, extracting principal components that encapsulate most of the data variance.
Feature Engineering
- Integration of Behavioral and Comfort Factors: Creating composite features that encapsulate daily activities, preferences, and comfort levels, offering a comprehensive understanding of the indoor environment.
- Calculation of Averages: Computing averages for behaviors like cooking frequency and window opening hours for a more concise feature representation.
- Merging Datasets: Combining housing price data with, IAQ parameters, and home characteristics based on common identifiers such as ZIP codes and Home IDs to provide a complete view of each home, including its market valuation.
Categorization of Prices: Homes are categorized into classes based on their total prices, simplifying the representation for classification models.

Model Training and Evaluation

- RandomForest Classifier: Employed for classifying homes into different classes using various features, with performance enhancement via GridSearchCV.

- Feature Importance Analysis: Using RandomForest classifiers to assess the importance of different features, which aids in retaining influential features for classification tasks.

- Regression Models: Implementing RandomForestRegressor and GradientBoostingRegressor to predict air quality metrics like PM2.5, CO2, NO2, Temperature, and Relative Humidity.

Advanced Techniques and Visualization

- High-Dimensional Data Visualization: Applying techniques like t-SNE to visualize complex data, aiding in pattern and cluster identification.

- SMOTE Application: Utilizing Synthetic Minority Over-sampling Technique to balance the dataset for improved model performance.

- GridSearchCV for Hyperparameter Tuning: Optimizing RandomForest classifiers to enhance model performance.

- Residual Analysis: Conducting residual analysis in regression models to evaluate the fit and detect patterns.
Rationale Behind the Approach

- Comprehensive Data Analysis: By integrating data cleaning, feature engineering, and model training, these scripts offer a thorough approach to understanding the factors that influence indoor air quality and home classification.
Focus on Feature Relevance: The emphasis on feature importance helps in building more efficient and interpretable models.

- Adaptability and Scalability: The scripts are designed to be adaptable to different datasets and scalable for larger datasets or more complex models.

Usage
To use these scripts:

- Install all necessary Python libraries.
- Run the data cleaning and preprocessing steps to prepare your dataset.
- Engage in feature engineering to develop a robust set of features for analysis.
- Train models using the processed data and evaluate their performance.

Contributions
Enhancements in data preprocessing, feature engineering methods, or experimenting with different modeling techniques to improve prediction accuracy and reliability are welcome.
