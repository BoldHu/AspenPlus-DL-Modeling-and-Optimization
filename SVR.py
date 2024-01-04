# using SVR to predict the sum of aromatics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# read the data
data = pd.read_csv('data/origin_data.csv')
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
regressor = SVR(kernel='rbf', C=1, epsilon=0.01, gamma='scale', shrinking=True)
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
plt.plot([0.63, 0.67], [0.63, 0.67], 'r')
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.title('SVR')
plt.savefig('figures/SVR_origin.png')
plt.show()

# save the model
import pickle
pickle.dump(regressor, open('checkpoint/SVR_origin.pkl', 'wb'))