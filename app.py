import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Config
st.set_page_config(page_title="QA Strategy Dashboard", layout="wide")

# Load data and pipeline model
df = pd.read_csv("data/cleaned_qa_data.csv")
model = joblib.load("model/churn_model_pipeline.pkl")

# Title
st.title("📊 AI-Powered QA Strategy Dashboard")
st.markdown("Created by **Billgates D** – Data Analyst")

# KPIs
st.subheader("📈 Company KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Total Projects", len(df))
col2.metric("Avg Revenue", f"₹{df['Revenue_Generated_INR'].mean():,.0f}")
col3.metric("Renewal Rate", f"{df['Renewed_Flag'].mean() * 100:.1f}%")

# 🌍 Revenue by Country
st.subheader("🌍 Revenue by Country")
country_revenue = df.groupby("Country")["Revenue_Generated_INR"].sum().reset_index()

fig = px.choropleth(
    country_revenue,
    locations="Country",
    locationmode="country names",
    color="Revenue_Generated_INR",
    color_continuous_scale="greens",
    title="💰 Total Revenue by Country"
)
st.plotly_chart(fig, use_container_width=True)

# 🔮 Predict Client Churn
st.subheader("🔮 Predict Client Renewal Likelihood")
uploaded_file = st.file_uploader("📤 Upload Client CSV (raw format)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    st.write("📋 Uploaded Data Preview:")
    st.dataframe(input_df)

    try:
        # Make prediction
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)

        input_df["Predicted_Renewal"] = prediction
        input_df["Renewal_Confidence (%)"] = (proba[:, 1] * 100).round(2)


        st.success("✅ Prediction successful!")
        st.write("🔍 Prediction Results:")
        st.dataframe(input_df)

        st.download_button(
            label="📥 Download Predictions",
            data=input_df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_clients.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Check that your input CSV has the required feature columns.")

# 📊 Feature Importance
st.subheader("📊 Feature Importance – What Drives Renewal?")

# Only works if model has feature_importances_
if hasattr(model.named_steps["classifier"], "feature_importances_"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Get feature names
    cat_cols = ["Industry", "Country", "Service_Used", "Project_Size", "Project_Month"]
    num_cols = [
        "Project_Duration_Days", "Project_Cost_INR", "Revenue_Generated_INR",
        "Team_Size", "Feedback_Score", "Profit_INR", "ROI", "Client_Success_Score"
    ]
    
    # OneHotEncoder gets feature names
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cols = ohe.get_feature_names_out(cat_cols)
    all_features = list(encoded_cols) + num_cols

    # Get importances
    importances = model.named_steps["classifier"].feature_importances_
    top_idx = np.argsort(importances)[-15:][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(all_features)[top_idx][::-1], importances[top_idx][::-1], color="skyblue")
    plt.xlabel("Importance Score")
    plt.title("Top 15 Feature Importances")
    st.pyplot(plt)
else:
    st.warning("⚠️ Feature importance is not available for this model.")

# 💡 Strategic Insights Section
st.subheader("💡 Auto-Generated Strategic Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    # Average renewal rate by industry
    industry_renew = df.groupby("Industry")["Renewed_Flag"].mean().sort_values(ascending=False)
    st.markdown("**🔍 Industries with highest renewal rates:**")
    st.dataframe(industry_renew.head(5).apply(lambda x: f"{x * 100:.2f}%"))

    # ROI vs Project Size
    roi_by_size = df.groupby("Project_Size")["ROI"].mean().sort_values(ascending=False)
    st.markdown("**💰 ROI by Project Size:**")
    st.dataframe(roi_by_size.round(2))

with insight_col2:
    # Renewal rate by Country
    country_renew = df.groupby("Country")["Renewed_Flag"].mean().sort_values(ascending=False)
    st.markdown("**🌍 Countries with highest renewal rates:**")
    st.dataframe(country_renew.head(5).apply(lambda x: f"{x * 100:.2f}%"))

    # Profit by Feedback Level
    profit_by_feedback = df.groupby("Feedback_Level")["Profit_INR"].mean().sort_values(ascending=False)
    st.markdown("**📈 Avg Profit by Feedback Level:**")
    st.dataframe(profit_by_feedback.round(2))

# 🌍 Interactive World Map with Filters
st.subheader("🌐 Explore Global QA Projects")

# Sidebar filters
st.sidebar.header("🌎 Map Filters")
selected_country = st.sidebar.multiselect("Filter by Country", df["Country"].unique())
selected_industry = st.sidebar.multiselect("Filter by Industry", df["Industry"].unique())
selected_service = st.sidebar.multiselect("Filter by Service", df["Service_Used"].unique())

# Apply filters
filtered_df = df.copy()
if selected_country:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_country)]
if selected_industry:
    filtered_df = filtered_df[filtered_df["Industry"].isin(selected_industry)]
if selected_service:
    filtered_df = filtered_df[filtered_df["Service_Used"].isin(selected_service)]

# Aggregate for map
map_df = filtered_df.groupby("Country").agg({
    "Revenue_Generated_INR": "sum",
    "ROI": "mean",
    "Client_Name": "count"
}).reset_index().rename(columns={"Client_Name": "Projects_Count"})

# Plot with Plotly
fig = px.scatter_geo(
    map_df,
    locations="Country",
    locationmode="country names",
    size="Projects_Count",
    color="Revenue_Generated_INR",
    hover_name="Country",
    size_max=40,
    projection="natural earth",
    title="📌 Global QA Engagements: Revenue & Volume",
    template="plotly_dark",
    color_continuous_scale="Tealgrn"
)

fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

st.plotly_chart(fig, use_container_width=True)
