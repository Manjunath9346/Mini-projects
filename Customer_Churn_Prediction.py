import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\sanka\OneDrive\Desktop\Projects\ML_project\Telco-Customer-Churn.csv")  # Make sure path is correct
print("Data loaded. Shape:", df.shape)

# Step 2: Clean the data
df.dropna(inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Step 3: Encode categorical variables
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Step 4: Split the data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)  # Use max_depth to keep tree readable
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the Decision Tree using matplotlib
plt.figure(figsize=(40, 20))  # Large figure size for visibility
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Stayed", "Churned"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree - Customer Churn Prediction", fontsize=18)
plt.savefig("customer_churn_tree_plot.png", dpi=300, bbox_inches="tight")  # Optional: saves the image
plt.show()
