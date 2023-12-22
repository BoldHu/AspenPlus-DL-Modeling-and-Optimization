import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Read the data
data = pd.read_csv('data/result.csv')

# Split the data into training set and test set
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)

# Initialize the SVR model
regressor = SVR(kernel='rbf', C=10, epsilon=0.1, gamma=0.01, shrinking=True)

# Function to train the model on a subset of the data and calculate the score
def train_and_score(subset_frac):
    subset_size = int(subset_frac * len(X_train_scaled))
    regressor.fit(X_train_scaled[:subset_size], y_train[:subset_size])
    y_pred = regressor.predict(X_test_scaled)
    return r2_score(y_test, y_pred)

# Train the model on increasing fractions of the training set and track the scores
fractions = np.linspace(0.1, 1.0, 10)  # Adjust the number of steps as needed
scores = [train_and_score(frac) for frac in fractions]

# Plot the scores
plt.plot(fractions, scores, marker='o')
plt.xlabel('Fraction of Training Data Used')
plt.ylabel('R2 Score')
plt.title('SVR Performance Over Training Progress')
plt.savefig('figures/SVR_training_progress.png')
plt.show()
