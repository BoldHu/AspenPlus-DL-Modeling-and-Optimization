import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.PSO import PSO
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the SVR model with error handling
try:
    regressor = pickle.load(open('checkpoint/SVR.pkl', 'rb'))
except FileNotFoundError:
    print("File not found. Please check the path to the SVR model.")
    exit()

# Define the objective function for PSO
def objective_function(x):
    # Validate input dimensions or ranges if necessary
    if not (len(x) == 10 and all(lb[i] <= x[i] <= ub[i] for i in range(len(x)))):
        raise ValueError("Input dimensions or range are incorrect.")
    # convert the input by scaling
    x = sc_X.transform(np.array(x).reshape(1, -1))
    # predict the result
    y_pred = regressor.predict(x)
    # print(f"Predicted Yield: {y_pred[0]}")
    return -y_pred[0]

# Define the search space for PSO
lb = [470, 0, 470, 0, 470, 0, 470, 0, 900, 0.4] # lower bound
ub = [600, 1, 600, 1, 600, 1, 600, 1, 1100, 0.6] # upper bound

# read the data
data = pd.read_csv('data/result.csv')
X_train = data.iloc[:, :-1].values
y_train = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# scale the data
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = sc_y.transform(y_test.reshape(-1, 1)).ravel()

# get expection and variance of each parameter and modify the search space
# for i in range(len(data.columns)-1):
#     df = data.iloc[:, i]
#     E = df.mean()
#     V = df.var()
#     lb[i] = (lb[i] - E) / V
#     ub[i] = (ub[i] - E) / V

# Create a PSO optimizer with adjusted parameters (if needed)
pso = PSO(func=objective_function, n_dim=10, lb=lb, ub=ub, max_iter=100, pop=100, w=0.8, c1=0.5, c2=0.5)

# Run the optimization
best_params, best_loss = pso.run()
# convert the best_params to the original scale
# for i in range(len(data.columns)-1):
#     df = data.iloc[:, i]
#     E = df.mean()
#     V = df.var()
#     best_params[i] = best_params[i] * V + E
# print("Best Parameters:", best_params)
# print("Best Loss:", best_loss)

# convert the result to the original scale
best_loss = -best_loss
best_loss = sc_y.inverse_transform(np.array(best_loss).reshape(-1, 1)).ravel()
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
