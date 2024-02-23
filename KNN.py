import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.impute import SimpleImputer
# Read dataset from pandas
dataset = pd.read_csv("prices.csv")

#Removing null values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
columns_to_impute = ['open', 'close', 'low', 'high', 'volume']
dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])
#Correlation Matrix

import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = dataset.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(12, 8))
for i, (column, data) in enumerate(list(dataset.items())[2:], 1):
    plt.subplot(4, 2, i)
    sns.histplot(data, kde=True)
    plt.title(f'Histogram of {column}')

plt.tight_layout()
plt.show()

column_name = 'low'

plt.figure(figsize=(8, 6))

sns.boxplot(x=dataset[column_name], palette='Set2')

plt.title(f'Box Plot for {column_name}')
plt.show()

column_name = 'high'

plt.figure(figsize=(15, 6))

sns.boxplot(x=dataset[column_name], palette='Set2')

plt.title(f'Box Plot for {column_name}')
plt.show()

column_name = 'open'

plt.figure(figsize=(15, 6))

sns.boxplot(x=dataset[column_name], palette='Set2')

plt.title(f'Box Plot for {column_name}')
plt.show()

column_name = 'close'

plt.figure(figsize=(15, 6))

sns.boxplot(x=dataset[column_name], palette='Set2')

plt.title(f'Box Plot for {column_name}')
plt.show()

column_name = 'volume'

plt.figure(figsize=(15, 6))

sns.boxplot(x=dataset[column_name], palette='Set2')

plt.title(f'Box Plot for {column_name}')
plt.show()

numeric_columns = ['open', 'close', 'low', 'high', 'volume']

Q1 = dataset[numeric_columns].quantile(0.25)
Q3 = dataset[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

outliers_mask = ((dataset[numeric_columns] < (Q1 - 1.5 * IQR)) | (dataset[numeric_columns] > (Q3 + 1.5 * IQR)))

outliers_count = outliers_mask.sum()

print(outliers_count)

numeric_columns = ['open', 'close', 'low', 'high', 'volume']
Q1 = dataset[numeric_columns].quantile(0.25)
Q3 = dataset[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for column in numeric_columns:
    dataset[column] = np.clip(dataset[column], lower_bound[column], upper_bound[column])

numeric_columns = ['open', 'close', 'low', 'high', 'volume']

Q1 = dataset[numeric_columns].quantile(0.25)
Q3 = dataset[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

outliers_mask = ((dataset[numeric_columns] < (Q1 - 1.5 * IQR)) | (dataset[numeric_columns] > (Q3 + 1.5 * IQR)))

outliers_count = outliers_mask.sum()

print(outliers_count)

X = dataset[['open', 'high', 'low', 'volume']]
y = dataset['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['open', 'high', 'low', 'volume']])
X_test_scaled = scaler.transform(X_test[['open', 'high', 'low', 'volume']])

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)

from sklearn.metrics import r2_score

# Assuming you have already trained and defined your KNN regressor model
y_pred_regressor = knn_regressor.predict(X_test_scaled)

r2_knn_test = r2_score(y_test, y_pred_regressor) * 100  # Convert to percentage
print(f'KNN Regressor - R-squared (Coefficient of Determination) for Test Data: {r2_knn_test:.2f}%')

y_pred_train_regressor = knn_regressor.predict(X_train_scaled)
r2_knn_train = r2_score(y_train, y_pred_train_regressor) * 100  # Convert to percentage
print(f'KNN Regressor - R-squared (Coefficient of Determination) for Training Data: {r2_knn_train:.2f}%')

while True:
    open_val = float(input("Enter open value: "))
    low_val = float(input("Enter low value: "))
    high_val = float(input("Enter high value: "))
    volume_val = float(input("Enter volume value: "))

    user_input = pd.DataFrame([[open_val, high_val, low_val, volume_val]], columns=['open', 'high', 'low', 'volume'])
    user_input_scaled = scaler.transform(user_input)

    predicted_close_regressor = knn_regressor.predict(user_input_scaled)
    print("Regressor - Predicted close value:", predicted_close_regressor[0])

    choice = input("Do you want to make another prediction? (y/n): ")
    if choice.lower() != 'y':
        break

  y_class = np.where(y > y.median(), 1, 0)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_class, y_train_class)

y_pred_classifier = knn_classifier.predict(X_test_class)
accuracy_cls = accuracy_score(y_test_class, y_pred_classifier) * 100
precision_cls = precision_score(y_test_class, y_pred_classifier) * 100
recall_cls = recall_score(y_test_class, y_pred_classifier) * 100
f1_cls =f1_score(y_test_class, y_pred_classifier) * 100

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print(f'Classifier - Accuracy: {accuracy_cls:.2f}%')
print(f'Classifier - Precision: {precision_cls:.2f}%')
print(f'Classifier - Recall: {recall_cls:.2f}%')
print(f'Classifier - F1-Score: {f1_cls:.2f}%')
