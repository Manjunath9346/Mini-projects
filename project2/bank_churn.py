import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\sanka\OneDrive\Desktop\Projects\ML_project\bank-full.csv.csv", sep=';')

print("Data Loaded:", df.shape)
print(df.head())

# Step 2: Encode categorical variables
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Step 3: Split into features and target
X = df.drop('y', axis=1)   # 'y' is the target (0 = no, 1 = yes)
y = df['y']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
