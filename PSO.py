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

# find the best parameters for PSO

# Create a PSO optimizer with adjusted parameters (if needed) to find the best parameters to maximize the aromatics yield
pso = PSO(func=objective_function, n_dim=10, lb=lb, ub=ub, max_iter=30, pop=100, w=0.5, c1=1.5, c2=0.5)

# the initial parameters
print("Initial parameters:", pso.gbest_x)
print("Initial aromatics yield:", -pso.gbest_y)

# Run the optimization and save the original scale result
best_params, best_loss = pso.run()

# convert the result to the original scale
best_loss = -best_loss
print("Best aromatics yield:", best_loss)

# convert the parameters to the original scale
# best_params = sc_X.inverse_transform(np.array(best_params).reshape(1, -1))
print("Best parameters:", best_params)

# save the result
np.savetxt('data/PSO_params.csv', best_params, delimiter=',')
np.savetxt('data/PSO_loss.csv', best_loss, delimiter=',')

# Draw the convergence curve with enhancements
plt.plot(pso.gbest_y_hist)
plt.xlabel('Iteration')
plt.ylabel('Aromatics Yield')
plt.title('Convergence Curve')
plt.grid(True)
plt.savefig('figures/PSO.png')
plt.show()
