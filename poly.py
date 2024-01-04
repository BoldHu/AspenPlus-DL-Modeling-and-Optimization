import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#read the data
data = pd.read_csv('data/origin_data.csv')

# just use the data of 1,2 columns and three line
X = data.iloc[0:3, 0:2].values
print(X)
print(X.shape)

# standardize the data
# standardize the data
sc_X = StandardScaler()
X_std = sc_X.fit_transform(X)

# create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_std)

# print X_poly' shape
print(X_poly.shape)
print(X_poly)

# set column names
col_names = ['反应器1的温度(x1)', '反应器1的压强(x2)', 'x1^2', 'x1*x2', 'x2^2']
df = pd.DataFrame(X_poly, columns=col_names)

# save the data
df_std = pd.DataFrame(X_poly, columns=col_names)
df_std.to_csv('data/origin_data_std.csv', index=False)

