import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# ✅ Step 1: Load Dataset
data_path = "/kaggle/input/"
print(os.listdir(data_path))  # Check dataset folder

file_path = "/kaggle/input/prediction-data/Prediction_Data.xlsx"
data = pd.read_excel(file_path, sheet_name="vw_ChurnData")

# ✅ Step 2: Drop unnecessary columns
data = data.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)

# ✅ Step 3: Encode categorical columns
columns_to_encode = [
    'Gender', 'Married', 'State', 'Value_Deal', 'Phone_Service', 'Multiple_Lines',
    'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup',
    'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 'Streaming_Movies',
    'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing',
    'Payment_Method'
]

label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# ✅ Step 4: Encode target variable
data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})

# ✅ Step 5: Split data into training & testing sets
X = data.drop('Customer_Status', axis=1)
y = data['Customer_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 6: Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Step 7: Check Feature Names
print("Model trained with features:", X.columns.tolist())

# ✅ Step 8: Feature Importance Visualization
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(15, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Names')
plt.show()

# ✅ Step 9: Load new data for predictions
new_data = pd.read_excel(file_path, sheet_name="vw_JoinData")
original_data = new_data.copy()

# Keep Customer_ID for reference
customer_ids = new_data['Customer_ID']

# Drop unnecessary columns
new_data = new_data.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)

# Encode categorical variables
for column in new_data.select_dtypes(include=['object']).columns:
    new_data[column] = label_encoders[column].transform(new_data[column])

# ✅ Step 10: Predict churn and save results
new_predictions = rf_model.predict(new_data)
original_data['Customer_Status_Predicted'] = new_predictions

# Filter for churned customers
churned_customers = original_data[original_data['Customer_Status_Predicted'] == 1]
churned_customers.to_csv("/kaggle/working/Predictions.csv", index=False)

print("Predictions saved successfully!")

# ✅ Step 11: Interactive Web App using Kaggle Widgets
# Create Dropdowns and Sliders
gender_dropdown = widgets.Dropdown(options=label_encoders['Gender'].classes_, description="Gender:")
contract_dropdown = widgets.Dropdown(options=label_encoders['Contract'].classes_, description="Contract:")
internet_dropdown = widgets.Dropdown(options=label_encoders['Internet_Type'].classes_, description="Internet Type:")
monthly_slider = widgets.IntSlider(min=0, max=500, step=5, description="Monthly Charges:")
tenure_slider = widgets.IntSlider(min=0, max=72, step=1, description="Tenure (Months):")

# Prediction Button
predict_button = widgets.Button(description="Predict")
output_area = widgets.Output()

# ✅ Step 12: Define Prediction Function
def predict_churn(gender, contract, internet, monthly, tenure):
    # Convert categorical inputs using label encoders
    gender_encoded = label_encoders['Gender'].transform([gender])[0]
    contract_encoded = label_encoders['Contract'].transform([contract])[0]
    internet_encoded = label_encoders['Internet_Type'].transform([internet])[0]

    # Create DataFrame with correct column names
    input_data = pd.DataFrame([[gender_encoded, contract_encoded, internet_encoded, monthly, tenure]],
                              columns=['Gender', 'Contract', 'Internet_Type', 'Monthly_Charge', 'Tenure'])

    # Align features with training model
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1]  # Get churn probability

    return "Churn" if prediction == 1 else "Stay", round(probability * 100, 2)

# ✅ Step 13: Define Button Click Event
def on_button_click(b):
    with output_area:
        output_area.clear_output()
        result, probability = predict_churn(gender_dropdown.value, contract_dropdown.value, 
                                            internet_dropdown.value, monthly_slider.value, tenure_slider.value)
        print(f"🔹 **Prediction: {result}** (Churn Probability: {probability}%)")

predict_button.on_click(on_button_click)

# ✅ Step 14: Display Widgets
display(gender_dropdown, contract_dropdown, internet_dropdown, monthly_slider, tenure_slider, predict_button, output_area)
