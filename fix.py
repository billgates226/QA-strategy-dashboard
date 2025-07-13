import pandas as pd

# Load your file
df = pd.read_csv("data/cleaned_qa_data.csv")

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# Drop any duplicated columns
df = df.loc[:, ~df.columns.duplicated()]

# Reorder based on model training (example order — adjust if needed)
columns_order = [
    "Client_Name", "Industry", "Country", "Service_Used", "Project_Duration_Days",
    "Project_Cost_INR", "Revenue_Generated_INR", "Is_Renewed", "Team_Size",
    "Feedback_Score", "Project_Month", "Profit_INR", "ROI", "Project_Size",
    "Feedback_Level", "Renewed_Flag", "Client_Success_Score"
]

df = df[columns_order]

# Save cleaned file
df.head(10).to_csv("data/client_test_data_matched.csv", index=False)

print("✅ Clean test CSV ready!")
