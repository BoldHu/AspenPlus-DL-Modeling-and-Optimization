from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load your dataset
data = pd.read_csv('data/result.csv')

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Create an SVR model
regressor = SVR()

# Define a grid of hyperparameters to search
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.01, 1, 10],
    'epsilon': [0.01, 0.1],
    'gamma': ['scale', 'auto', 0.01, 1],
    'shrinking': [True, False]
}

# Create a GridSearchCV object with SVR and the parameter grid
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Extract the hyperparameters and R2 scores
results = pd.DataFrame(grid_search.cv_results_)
results = results[['param_kernel', 'param_C', 'param_epsilon', 'param_gamma', 'param_shrinking', 'mean_test_score']]

# Rename columns for clarity
results.columns = ['Kernel', 'C', 'Epsilon', 'Gamma', 'Shrinking', 'Mean R2 Score']

# Save to CSV
results.to_csv('data/hyperpara_search.csv', index=False)

print("Results have been saved to 'data/hyperpara_search.csv'.")
