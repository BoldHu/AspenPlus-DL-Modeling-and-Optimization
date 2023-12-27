# find the best kernal for SVR
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

# Load your dataset
data = pd.read_csv('data/origin_data.csv')

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
param_grid = {'kernel': ['linear', 'rbf', 'poly']}

# create different SVR models with different kernal
models = []
results = pd.DataFrame()
for kernal in param_grid['kernel']:
    model = SVR(kernel=kernal, C=10, epsilon=0.01, gamma='auto', shrinking=True)
    model.fit(X_train, y_train)
    models.append(model)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"The R2 score of {kernal} is {r2}")
    results = results.append({'param_kernel': kernal, 'mean_test_score': r2}, ignore_index=True)
    

results = results[['param_kernel', 'mean_test_score']]
results.columns = ['Kernel', 'Mean R2 Score']
results.to_csv('data/kernal_search.csv', index=False)
print("Results have been saved to 'data/kernal_search.csv'.")
import matplotlib.pyplot as plt
import seaborn as sns  # for better visual aesthetics

# Assuming results DataFrame is already defined and loaded
# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(10, 6))  # Adjust the figure size
barplot = sns.barplot(x='Kernel', y='Mean R2 Score', data=results, palette="viridis", edgecolor='black')

# Add labels to each bar
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), 
                     textcoords = 'offset points')

# Set labels and title
plt.xlabel('Kernel Name', fontsize=12)
plt.ylabel('Mean R2 Score', fontsize=12)
plt.title('SVR Kernel Performance Comparison', fontsize=14)

# Save the plot
plt.savefig('figures/kernel_search.png', bbox_inches='tight')  # Adjust for tight layout

# Show the plot
plt.show()
