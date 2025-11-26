import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the "train dataset"
df = pd.read_csv(r"C:\users\hp\Downloads\train.csv")

# Inspect the data
print("""Train Dataset: 
      """
      , df.head())

#set x and y

X = df.drop(['Hardness','id'], axis=1)
y = df['Hardness']

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=500, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
print(y_pred)
r2 = r2_score(y_test, y_pred)
print(f'R² Score for "train dataset": {r2:.4f}')

#Import "test dataset" 
new_test_data = pd.read_csv(r"C:\Users\hp\Downloads\test.csv")
print("Test Dataset: ", new_test_data.head())

#working on "test dataset"
newx = new_test_data.drop(['Hardness','id'], axis=1)
newy = new_test_data["Hardness"]

new_y_pred = rf.predict(newx)
print(new_y_pred)

#export to CSV
predictions_df = pd.DataFrame({'Predicted_Hardness': new_y_pred})
predictions_df.to_csv(r"C:\Users\hp\Downloads\predictions.csv", index=False)


