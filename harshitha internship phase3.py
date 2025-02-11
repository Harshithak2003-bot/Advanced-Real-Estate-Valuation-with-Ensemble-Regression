
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv('real_estate_165_data.csv')
print("Original DataFrame columns:", df.columns)

target_column = 'Property_Value'  

# Drop the target column to create features (X) and target (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

# Get numerical features from original X (before splitting)
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numerical features *after* train_test_split
categorical_features = ['Location', 'Property_Type']
categorical_features = [col for col in categorical_features if col in X_train.columns]


# Create a ColumnTransformer 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) 
    ])

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# ... (Model training and evaluation - same as before) ...
# Define hyperparameter grids
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
gb_param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01]}

# Perform GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)
rf_regressor = RandomForestRegressor(**rf_grid_search.best_params_, random_state=42)

# Perform GridSearchCV for Gradient Boosting
gb_grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_param_grid, cv=5)
gb_grid_search.fit(X_train, y_train)
gb_regressor = GradientBoostingRegressor(**gb_grid_search.best_params_, random_state=42)

# Train and evaluate individual models
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)
gb_regressor.fit(X_train, y_train)
gb_pred = gb_regressor.predict(X_test)

# Train and evaluate Stacking Regressor
stacking_regressor = StackingRegressor(estimators=[('rf', rf_regressor), ('gb', gb_regressor)],
                                       final_estimator=LinearRegression())
stacking_regressor.fit(X_train, y_train)
stacking_pred = stacking_regressor.predict(X_test)


# --- Model Performance Output ---
print("Random Forest Regressor MSE:", mean_squared_error(y_test, rf_pred))
print("Gradient Boosting Regressor MSE:", mean_squared_error(y_test, gb_pred))
print("Stacking Regressor MSE:", mean_squared_error(y_test, stacking_pred))

# --- Cross-Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = [] 
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}:")
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    # Preprocess the data for this fold
    X_train_fold = preprocessor.fit_transform(X_train_fold)
    X_test_fold = preprocessor.transform(X_test_fold)
    
    # Train and evaluate models for this fold 
    rf_regressor.fit(X_train_fold, y_train_fold)
    rf_pred_fold = rf_regressor.predict(X_test_fold)
    gb_regressor.fit(X_train_fold, y_train_fold)
    gb_pred_fold = gb_regressor.predict(X_test_fold)
    stacking_regressor.fit(X_train_fold, y_train_fold)
    stacking_pred_fold = stacking_regressor.predict(X_test_fold)
    
    # Calculate and store MSE scores for this fold
    mse_scores.append([
        mean_squared_error(y_test_fold, rf_pred_fold),
        mean_squared_error(y_test_fold, gb_pred_fold),
        mean_squared_error(y_test_fold, stacking_pred_fold)
    ])

    print("Random Forest Regressor MSE:", mse_scores[-1][0])
    print("Gradient Boosting Regressor MSE:", mse_scores[-1][1])
    print("Stacking Regressor MSE:", mse_scores[-1][2])
    

# --- Visualization Section ---

# 1. Bar chart (average MSE)
plt.figure(figsize=(8, 6))  
labels = ['RF', 'GB', 'Stacking']
avg_mse_scores = [sum(scores)/len(scores) for scores in zip(*mse_scores)]
plt.bar(labels, avg_mse_scores)
plt.xlabel('Model')
plt.ylabel('Average MSE')
plt.title('Model Performance Comparison (Average MSE)')
plt.show()

# 2. Line graph (cross-validation performance)
plt.figure(figsize=(8, 6))  
x = [i+1 for i in range(len(mse_scores))]
y_rf = [fold_scores[0] for fold_scores in mse_scores]
y_gb = [fold_scores[1] for fold_scores in mse_scores]
y_stacking = [fold_scores[2] for fold_scores in mse_scores]
plt.plot(x, y_rf, label='RF')
plt.plot(x, y_gb, label='GB')
plt.plot(x, y_stacking, label='Stacking')
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title('Cross-Validation Performance')
plt.legend()
plt.show()



# 4. Box plot (Property Value by Location)
plt.figure(figsize=(8, 6))  
sns.boxplot(x='Location', y=target_column, data=df)
plt.xlabel('Location')
plt.ylabel('Property Value')
plt.title('Property Value Distribution by Location')
plt.xticks(rotation=45, ha='right')  
plt.show()