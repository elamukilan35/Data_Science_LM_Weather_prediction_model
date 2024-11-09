import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", None)
pd.set_option("display.max_rows", None)

# Load data
data = pd.read_csv("weather_data.csv", index_col='Date_Time')
print(data.nunique())
print(data.dtypes)
print(data.head())
print(len(data))
print("Identify null values:", data.isnull().sum())

# Reduce data size
reduced_data = data.sample(frac=0.09).reset_index(drop=True)
print("Length of the reduced data:", len(reduced_data))
print("Shape of the reduced data:", reduced_data.shape)


# Assuming we want to visualize the temperature over time
plt.figure(figsize=(8, 4))
plt.plot(reduced_data.index, reduced_data['Temperature_C'], marker='o', linestyle='-', color='b')
plt.xlabel('Index')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Over Time')
plt.grid(True)
plt.show()
#
# Handle categorical columns
le = LabelEncoder()
reduced_data['Location'] = le.fit_transform(reduced_data['Location'])
print("Label encoded 'Location':", reduced_data['Location'].value_counts())

# Define features and target
X = reduced_data.drop(['Temperature_C'], axis=1)
y = reduced_data['Temperature_C']
cat_col = [col for col in reduced_data.columns if reduced_data[col].dtype == 'object']
ohe = OneHotEncoder(sparse_output=False)
cat_to_num_col = ohe.fit_transform(reduced_data[cat_col])
encoded_data = pd.DataFrame(cat_to_num_col, columns=ohe.get_feature_names_out())
reduced_data.drop(cat_col, axis=1, inplace=True)
new_data = pd.concat([reduced_data, encoded_data], axis=1)
print("New_data Frame: \n", new_data)

# Example visualization: Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(new_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# Example visualization: Pairplot
sns.pairplot(new_data)
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(new_data, y, test_size=0.2, random_state=42)
print("Training Features:", X_train.columns)
print("Test Features:", X_test.columns)

# Random Forest Regressor
rf_model = RandomForestRegressor(max_depth=5, min_samples_leaf=1, min_samples_split=2, n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_train_pred_rf = rf_model.predict(X_train)
r2_train_rf = r2_score(y_train, y_train_pred_rf)
y_test_pred_rf = rf_model.predict(X_test)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print("RandomForest R2 Score of training data:", r2_train_rf)
print("RandomForest R2 Score of test data:", r2_test_rf)

# Plot training data predictions vs actual
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred_rf, alpha=0.5, color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Training Data')

# Plot test data predictions vs actual
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_rf, alpha=0.5, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Data')

# Display the plots
plt.tight_layout()
plt.show()

# SVM model
svr_model = SVR(kernel='rbf', C=1, epsilon=0.1)
svr_model.fit(X_train, y_train)
y_train_pred_svr = svr_model.predict(X_train)
r2_train_svr = r2_score(y_train, y_train_pred_svr)
y_test_pred_svr = svr_model.predict(X_test)
r2_test_svr = r2_score(y_test, y_test_pred_svr)
print("SVR R2 Score of training data:", r2_train_svr)
print("SVR R2 Score of test data:", r2_test_svr)

# Plot training and test data predictions
plt.figure(figsize=(8, 4))

# Training data
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred_svr, alpha=0.5, color='blue')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Training Data')

# Test data
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_svr, alpha=0.5, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Data')

# Display the plots
plt.tight_layout()
plt.show()

# Cross-validation with RandomForestRegressor
cv_scores = cross_val_score(rf_model, new_data, y, cv=5, scoring='r2')
print("RandomForest Cross-Validation Scores:", cv_scores)
print("Mean RandomForest Cross-Validation R2 Score:", np.mean(cv_scores))

# Plotting cross-validation scores
plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-', color='b', label='Cross-Validation R2 Scores')
plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', label='Mean R2 Score')
plt.xlabel('Fold Number')
plt.ylabel('R2 Score')
plt.title('Cross-Validation Scores for RandomForestRegressor')
plt.legend()
plt.grid(True)
plt.show()

# Assuming new_data is your DataFrame with the transformed features
retrieved_index = new_data.reset_index()
print(retrieved_index.head())
