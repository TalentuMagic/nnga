import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


# Load the dataset
dataset_path = './forestfires.csv'
data = pd.read_csv(dataset_path)

# Remove non-numeric columns that are not useful for regression
data = data.drop(['month', 'day'], axis=1)

# Convert categorical variables to numeric using LabelEncoder
le = LabelEncoder()
data['area'] = le.fit_transform(data['area'])

# Prepare the data
X = data.drop(['area'], axis=1)  # Features (excluding 'area' column)
y = np.log1p(data['area'])  # Target variable ; log transform - skewed distribution of the target variable to improve model fit

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale & Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to apply Decision Tree Regressor
def apply_decision_tree_regressor(X_train, X_test, y_train, y_test):
    # Define the parameter grid for grid search
    param_grid = {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create and fit the Decision Tree Regressor model
    dt_regressor = DecisionTreeRegressor(random_state=42)
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(dt_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best model with the optimal hyperparameters
    best_model = grid_search.best_estimator_
    
    # Predict on the test set
    y_pred = best_model.predict(X_test)
    
    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Decision Tree Regressor -> Mean Squared Error:", mse)
    print("Decision Tree Regressor -> R-Squared:", r2)

# Function to apply Voting Regressor
def apply_voting_regressor(X_train, X_test, y_train, y_test):
    # Create individual regressor models
    dt_regressor = DecisionTreeRegressor(random_state=42)
    mlp_regressor = MLPRegressor(random_state=42)
    
    # Create the Voting Regressor model
    voting_regressor = VotingRegressor(estimators=[('dt', dt_regressor), ('mlp', mlp_regressor)])
    
    # Fit the Voting Regressor model
    voting_regressor.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = voting_regressor.predict(X_test)
    
    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Voting Regressor -> Mean Squared Error:", mse)
    print("Voting Regressor -> R-Squared:", r2)

# Function to apply MLP Regressor
def apply_mlp_regressor(X_train, X_test, y_train, y_test):
    # Create and fit the MLP Regressor model
    mlp_regressor = MLPRegressor(random_state=42)
    mlp_regressor.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = mlp_regressor.predict(X_test)
    
    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MLP Regressor -> Mean Squared Error:", mse)
    print("MLP Regressor -> R-Squared:", r2)

# Function to apply Keras regression
def apply_keras_regression(X_train, X_test, y_train, y_test):
    # Create the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=64, batch_size=32, verbose=1)

    # Predict the model
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = model.evaluate(X_test, y_test, verbose=0)
    r2 = r2_score(y_test, y_pred)
    print("Keras Regression -> Mean Squared Error:", mse)
    print("Keras Regression -> R-squared:", r2)

# Function to apply Random Forest Regression
def apply_random_forest_regression(X_train, X_test, y_train, y_test):
    # Define the parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create the Random Forest Regressor model
    rf_regressor = RandomForestRegressor(random_state=42)
    
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best model with the optimal hyperparameters
    best_model = grid_search.best_estimator_
    
    # Predict on the test set
    y_pred = best_model.predict(X_test)
    
    # Calculate and print the mean squared error and R-squared
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Random Forest Regressor -> Mean Squared Error:", mse)
    print("Random Forest Regressor -> R-Squared:", r2)


# Function to apply Support Vector Regression with hyperparameter tuning
def apply_svr(X_train, X_test, y_train, y_test):
    # Define the parameter grid for SVR
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }

    # Create and fit the SVR model with GridSearchCV
    svr_regressor = SVR()
    grid_search = GridSearchCV(svr_regressor, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best SVR model
    best_svr = grid_search.best_estimator_

    # Predict on the test set using the best model
    y_pred = best_svr.predict(X_test)

    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Support Vector Regression -> Mean Squared Error:", mse)
    print("Support Vector Regression -> R-Squared:", r2)


# Function to apply Gradient Boosting Regression with hyperparameter tuning
def apply_gradient_boosting_regressor(X_train, X_test, y_train, y_test):
    # Define the parameter grid for Gradient Boosting Regression
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5]
    }

    # Create and fit the Gradient Boosting Regression model with GridSearchCV
    gb_regressor = GradientBoostingRegressor()
    grid_search = GridSearchCV(gb_regressor, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best Gradient Boosting model
    best_gb = grid_search.best_estimator_

    # Predict on the test set using the best model
    y_pred = best_gb.predict(X_test)

    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Gradient Boosting Regression -> Mean Squared Error:", mse)
    print("Gradient Boosting Regression -> R-Squared:", r2)

# Apply the models
apply_keras_regression(X_train_scaled, X_test_scaled, y_train, y_test)
apply_svr(X_train_scaled, X_test_scaled, y_train, y_test)
apply_gradient_boosting_regressor(X_train, X_test, y_train, y_test)
apply_decision_tree_regressor(X_train, X_test, y_train, y_test)
apply_voting_regressor(X_train, X_test, y_train, y_test)
apply_mlp_regressor(X_train, X_test, y_train, y_test)
apply_random_forest_regression(X_train, X_test, y_train, y_test)