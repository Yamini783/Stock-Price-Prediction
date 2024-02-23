import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pickle

# Read dataset from pandas
dataset = pd.read_csv("prices.csv")

# Impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
columns_to_impute = ['open', 'close', 'low', 'high', 'volume']
dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])

# Remove outliers
numeric_columns = ['open', 'close', 'low', 'high', 'volume']
Q1 = dataset[numeric_columns].quantile(0.25)
Q3 = dataset[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
for column in numeric_columns:
    dataset[column] = dataset[column].clip(lower=lower_bound[column], upper=upper_bound[column])

# Split data into train and test sets for regression
X = dataset[['open', 'high', 'low', 'volume']]
y = dataset['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train_scaled, y_train)

# Split data into train and test sets for classification
y_class = np.where(y > y.median(), 1, 0)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Scale features for classification
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

# Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_class_scaled, y_train_class)

# Serialize DecisionTreeRegressor
with open('dt_regressor.pkl', 'wb') as file:
    pickle.dump(dt_regressor, file)

# Serialize DecisionTreeClassifier
with open('dt_classifier.pkl', 'wb') as file:
    pickle.dump(dt_classifier, file)

# Serialize the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
