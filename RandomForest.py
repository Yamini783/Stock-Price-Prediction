import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
random_forest_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
random_forest_regressor.fit(X_train_scaled, y_train)

# Create Random Forest Classifier model
y_class = np.where(y > y.median(), 1, 0)  # Convert to binary classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

random_forest_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
random_forest_classifier.fit(X_train_class, y_train_class)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict using the models
y_pred_regressor = random_forest_regressor.predict(X_test_scaled)
mae_rf = mean_absolute_error(y_test, y_pred_regressor)
mse_rf = mean_squared_error(y_test, y_pred_regressor)

mae_rf_percentage = mae_rf * 100
mse_rf_percentage = mse_rf * 100

print(f'Random Forest Regressor - Mean Absolute Error (MAE): {mae_rf_percentage:.2f}%')
print(f'Random Forest Regressor - Mean Squared Error (MSE): {mse_rf_percentage:.2f}%')

r2_rf_test = r2_score(y_test,y_pred_regressor) * 100  # Convert to percentage
print(f'Random Forest Regressor - R-squared (Coefficient of Determination) for Test Data: {r2_rf_test:.2f}%')

y_pred_train_regressor = random_forest_regressor.predict(X_train_scaled)
r2_rf_train = r2_score(y_train, y_pred_train_regressor) * 100  # Convert to percentage
print(f'Random Forest Regressor - R-squared (Coefficient of Determination) for Training Data: {r2_rf_train:.2f}%')


# User input for prediction
while True:
    open_val = float(input("Enter open value: "))
    low_val = float(input("Enter low value: "))
    high_val = float(input("Enter high value: "))
    volume_val = float(input("Enter volume value: "))


    user_input = pd.DataFrame([[open_val, high_val, low_val, volume_val]], columns=['open', 'high', 'low', 'volume'])
    user_input_scaled = scaler.transform(user_input)

    predicted_close_regressor = random_forest_regressor.predict(user_input_scaled)
    print("Regressor - Predicted close value:", predicted_close_regressor[0])


    predicted_class = random_forest_classifier.predict(user_input)
    print("Classifier - Predicted class:", predicted_class[0])

    choice = input("Do you want to make another prediction? (y/n): ")
    if choice.lower() != 'y':
        break


y_pred_classifier = random_forest_classifier.predict(X_test_class)

# Calculate accuracy and convert to percentage with two decimal places
accuracy_cls = accuracy_score(y_test_class, y_pred_classifier) * 100
print(f'Classifier - Accuracy: {accuracy_cls:.2f}%')

# Calculate precision and convert to percentage with two decimal places
precision_cls = precision_score(y_test_class, y_pred_classifier) * 100
print(f'Classifier - Precision: {precision_cls:.2f}%')

# Calculate recall and convert to percentage with two decimal places
recall_cls = recall_score(y_test_class, y_pred_classifier) * 100
print(f'Classifier - Recall: {recall_cls:.2f}%')

# Calculate F1-score and convert to percentage with two decimal places
f1_cls = f1_score(y_test_class, y_pred_classifier) * 100
print(f'Classifier - F1-score: {f1_cls:.2f}%')
