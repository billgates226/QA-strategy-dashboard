import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Load your cleaned data
df = pd.read_csv("data/cleaned_qa_data.csv")

# Define input features and target
features = [
    "Industry", "Country", "Service_Used", "Project_Duration_Days", "Project_Cost_INR",
    "Revenue_Generated_INR", "Team_Size", "Feedback_Score", "Project_Month",
    "Profit_INR", "ROI", "Project_Size", "Client_Success_Score"
]
target = "Renewed_Flag"

X = df[features]
y = df[target]

# Define categorical and numerical columns
categorical = ["Industry", "Country", "Service_Used", "Project_Size", "Project_Month"]
numerical = [col for col in features if col not in categorical]

# Preprocessing for categorical features
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

# Build pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the pipeline
pipeline.fit(X, y)

# Save the trained pipeline model
joblib.dump(pipeline, "model/churn_model_pipeline.pkl")
print("✅ Model pipeline saved to model/churn_model_pipeline.pkl")
