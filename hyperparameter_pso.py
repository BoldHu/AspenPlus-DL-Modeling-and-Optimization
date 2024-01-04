import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.PSO import PSO
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


# Load the SVR model with error handling
try:
    regressor = pickle.load(open('checkpoint/RF_best.pkl', 'rb'))
except FileNotFoundError:
    print("File not found. Please check the path to the SVR model.")
    exit()

# Define the objective function for PSO
def objective_function(x):
    # Validate input dimensions or ranges if necessary
    if not (len(x) == 10 and all(lb[i] <= x[i] <= ub[i] for i in range(len(x)))):
        raise ValueError("Input dimensions or range are incorrect.")
    # convert the input by scaling
    x = poly.fit_transform(np.array(x).reshape(1, -1))
    x = sc_X.transform(x)
    # predict the result
    y_pred = regressor.predict(x)
    return -y_pred[0]

# Define the search space for PSO
lb = [508, 0.3325, 508, 0.3325, 508, 0.3325, 508, 0.3325, 1013.84, 0.4] # lower bound
ub = [525, 0.3675, 525, 0.3675, 527, 0.3675, 527, 0.3675, 1120.56, 0.6] # upper bound

# read the data
data = pd.read_csv('data/origin_data.csv')
X_train = data.iloc[:, :-1].values
y_train = data.iloc[:, -1].values
# poly
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train = poly.fit_transform(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# scale the data
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Define a grid of hyperparameters
w_values = [0.5, 0.7, 0.9]
c1_values = [0.5, 1.0, 1.5]
c2_values = [0.5, 1.0, 1.5]
max_iter_values = [20]
pop_values = [50, 100]

# Initialize variables to store the best performance and corresponding parameters
best_performance = float('inf')
best_hyperparameters = {}

# Iterate over each combination of hyperparameters
for w in w_values:
    for c1 in c1_values:
        for c2 in c2_values:
            for max_iter in max_iter_values:
                for pop in pop_values:
                    # Run PSO with the current set of hyperparameters
                    pso = PSO(func=objective_function, n_dim=10, lb=lb, ub=ub, 
                              max_iter=max_iter, pop=pop, w=w, c1=c1, c2=c2)
                    _, loss = pso.run()
                    print("Current Performance:", loss)

                    # If the current performance is better, update the best performance and hyperparameters
                    if loss < best_performance:
                        best_performance = loss
                        best_hyperparameters = {'w': w, 'c1': c1, 'c2': c2, 'max_iter': max_iter, 'pop': pop}

# Print the best hyperparameters and performance
print("Best Hyperparameters:", best_hyperparameters)
print("Best Performance:", best_performance)
