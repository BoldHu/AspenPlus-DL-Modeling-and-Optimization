# using SVR to predict the sum of aromatics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import random

# read the data
data = pd.read_csv('data/result.csv')
# split the data into training set and test set
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_train.head())

# scale the data
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# train the model
regressor = SVR(kernel='rbf', C=10, epsilon=0.01, gamma=0.01, shrinking=True)
regressor.fit(X_train, y_train)

# predict the result and convert the result to the original scale
y_pred = regressor.predict(X_test)

# calculate the R2 score
r2 = r2_score(y_test, y_pred)
print('The R2 score is ' + str(r2))

# save the y_test and y_pred to .csv file
y_test = pd.DataFrame(y_test)
y_pred = pd.DataFrame(y_pred)
y_test.to_csv('data/y_test.csv')
y_pred.to_csv('data/y_pred.csv')

# draw the result y_test is black and y_pred is blue
plt.scatter(y_test, y_pred, color='blue')
# draw the line y=x
plt.plot([0.5, 0.75], [0.5, 0.75], 'r')
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.title('SVR')
plt.savefig('figures/SVR.png')
plt.show()

# save the model
import pickle
pickle.dump(regressor, open('checkpoint/SVR.pkl', 'wb'))

# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load your dataset
# data = pd.read_csv('data/result.csv')

# # Split the data into features and target
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)


# # Create an SVR model
# regressor = SVR()

# # Define a smaller grid of hyperparameters to search
# param_grid = {
#     'kernel': ['linear', 'rbf'],
#     'C': [0.01, 1, 10],
#     'epsilon': [0.01, 0.1],
#     'gamma': ['scale', 'auto', 0.01, 1],
#     'shrinking': [True, False]
# }

# # Create a GridSearchCV object with SVR and the parameter grid
# grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)

# # Get the best model with the best hyperparameters
# best_model = grid_search.best_estimator_

# # Evaluate the best model on the test data
# y_pred = best_model.predict(X_test)

# # You can now evaluate the performance of the best model using appropriate metrics


