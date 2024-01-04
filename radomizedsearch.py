from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns  # for better visual aesthetics

# Load your dataset
data = pd.read_csv('data/origin_data.csv')

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Scale the features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Create an SVR model
svr_regressor = SVR()

# Define a grid of hyperparameters to search
param_grid_svr = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'shrinking': [True, False]
}

# Create a GridSearchCV object with SVR and the parameter grid
grid_search_svr = GridSearchCV(estimator=svr_regressor, param_grid=param_grid_svr, cv=5, n_jobs=-1, verbose=2)
grid_search_svr.fit(X_train, y_train)

# the best parameters for SVR
print(grid_search_svr.best_params_)
print(grid_search_svr.best_score_)

# Create a RandomForest model
rf_regressor = RandomForestRegressor()

# Define a grid of hyperparameters to search for RandomForest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create a GridSearchCV object with RandomForest and the parameter grid
grid_search_rf = GridSearchCV(estimator=rf_regressor, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

# the best parameters for RandomForest
print(grid_search_rf.best_params_)
print(grid_search_rf.best_score_)

# Choose the best model based on cross-validation score
if grid_search_svr.best_score_ > grid_search_rf.best_score_:
    best_regressor = grid_search_svr.best_estimator_
else:
    best_regressor = grid_search_rf.best_estimator_

# Predict and calculate R2 score
y_pred = best_regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('The R2 score is:', r2)
print('The MSE is:', mse)

# Plot true vs predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.title(type(best_regressor).__name__)
# save
plt.savefig('figures/SVR_best.png')
plt.show()

# Plot residuals
residuals = y_test - y_pred
# Use seaborn's residual plot feature which automatically plots the regression line as well
sns.residplot(x=y_pred, y=residuals, lowess=True, color="g", 
              scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 2})

plt.xlabel('Predicted value')
plt.ylabel('Residuals')
plt.title('Residual Plot with Lowess Smoothing')
plt.grid(True)
plt.show()

# save the best model
import pickle
pickle.dump(best_regressor, open('checkpoint/RF_best.pkl', 'wb'))