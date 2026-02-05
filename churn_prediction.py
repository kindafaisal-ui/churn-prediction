import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# STEP 1: LOAD AND EXPLORE THE DATA
# ==========================================
# Loading the Telco Customer Churn dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("--- STEP 1: DATA EXPLORATION ---")
print(f"Dataset Shape: {df.shape}")
print("\n--- Data Types ---\n", df.dtypes)
print("\n--- Missing Values ---\n", df.isnull().sum())
print("\n--- Target Distribution (Churn) ---\n", df['Churn'].value_counts())

# Visualization for Step 1
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribution of Customer Churn')
# TAKE YOUR SCREENSHOT OF THE CHART NOW!
plt.show() 

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================
# 1. Handle missing values in TotalCharges (convert spaces to NaN then drop)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# 2. Select relevant features (dropping unique ID)
df = df.drop('customerID', axis=1)

# 3. Convert categorical variables to numeric using One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Separate features (X) from target (y)
# After encoding, 'Churn_Yes' is our target (1=Churn, 0=No Churn)
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# ==========================================
# STEP 3: SPLIT THE DATA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# STEP 4: TRAIN KNN MODEL
# ==========================================
# Feature Scaling (Crucial for KNN performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Start with n_neighbors=5 as requested
k_value = 5
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)

# ==========================================
# STEP 5: EVALUATE THE MODEL
# ==========================================
y_pred = knn.predict(X_test)

print(f"\n--- STEP 5: EVALUATION RESULTS (K={k_value}) ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# STEP 6: EXPERIMENT WITH DIFFERENT K VALUES
# ==========================================
print("\n--- STEP 6: EXPERIMENTING WITH DIFFERENT K VALUES ---")
k_list = [1, 3, 5, 7, 9, 11, 15]
for k in k_list:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"K = {k:2d} | Accuracy: {score:.4f}")