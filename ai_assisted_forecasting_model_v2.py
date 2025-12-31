import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fintech AI",
    layout="wide"
)

st.title("📊 Fintech AI")
st.caption("AI assisted forecasting model")

# ---------------- Sidebar ----------------
st.sidebar.header("📂 Upload Financial Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

forecast_months = st.sidebar.slider(
    "Forecast Horizon (Months)", 3, 24, 12
)

use_seasonality = st.sidebar.checkbox(
    "Include Seasonality (Month-wise)", value=True
)

st.sidebar.subheader("🎛 Forecast Drivers")

revenue_driver = st.sidebar.slider("Revenue Adjustment (%)", -20, 20, 0)
variable_cost_driver = st.sidebar.slider("Variable Cost Adjustment (%)", -20, 20, 0)
fixed_cost_driver = st.sidebar.slider("Fixed Cost Adjustment (%)", 0, 20, 0)

st.sidebar.subheader("📌 Scenario Selection")

c1, c2, c3 = st.sidebar.columns(3)

if c1.button("Base"):
    revenue_driver, variable_cost_driver, fixed_cost_driver = 0, 0, 0

if c2.button("Optimistic"):
    revenue_driver, variable_cost_driver, fixed_cost_driver = 10, -5, 2

if c3.button("Pessimistic"):
    revenue_driver, variable_cost_driver, fixed_cost_driver = -10, 10, 5

st.info(
    f"""
    **Active Drivers**
    - Revenue: {revenue_driver}%
    - Variable Cost: {variable_cost_driver}%
    - Fixed Cost: {fixed_cost_driver}%
    - Seasonality: {"ON" if use_seasonality else "OFF"}
    """
)

# ---------------- Main Logic ----------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values('Month').reset_index(drop=True)

    st.subheader("📊 Uploaded Data")
    st.dataframe(df)

    df['t'] = np.arange(len(df))
    df['Total_Cost'] = df['Fixed_Cost'] + df['Variable_Cost']

    # --------- FEATURE ENGINEERING ----------
    if use_seasonality:
        df['Month_Num'] = df['Month'].dt.month
        month_dummies = pd.get_dummies(df['Month_Num'], prefix='M', drop_first=True)
        X = pd.concat([df[['t']], month_dummies], axis=1)
    else:
        X = df[['t']]

    # --------- REVENUE MODEL ----------
    revenue_model = LinearRegression()
    revenue_model.fit(X, df['Revenue'])
    df['Revenue_Forecast'] = revenue_model.predict(X)

    # --------- COST MODEL ----------
    cost_model = LinearRegression()
    cost_model.fit(X, df['Total_Cost'])
    df['Cost_Forecast'] = cost_model.predict(X)

    # --------- FUTURE DATA ----------
    future_t = np.arange(len(df), len(df) + forecast_months)
    future_months = pd.date_range(
        df['Month'].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq='MS'
    )

    future_df = pd.DataFrame({'Month': future_months, 't': future_t})

    if use_seasonality:
        future_df['Month_Num'] = future_df['Month'].dt.month
        future_dummies = pd.get_dummies(
            future_df['Month_Num'], prefix='M', drop_first=True
        )
        future_dummies = future_dummies.reindex(
            columns=month_dummies.columns, fill_value=0
        )
        X_future = pd.concat([future_df[['t']], future_dummies], axis=1)
    else:
        X_future = future_df[['t']]

    future_df['Revenue_Forecast'] = revenue_model.predict(X_future)
    future_df['Cost_Forecast'] = cost_model.predict(X_future)

    # --------- DRIVER ADJUSTMENTS ----------
    rev_factor = 1 + revenue_driver / 100
    cost_factor = 1 + (variable_cost_driver + fixed_cost_driver) / 200

    df['Adj_Revenue'] = df['Revenue_Forecast'] * rev_factor
    future_df['Adj_Revenue'] = future_df['Revenue_Forecast'] * rev_factor

    df['Adj_Cost'] = df['Cost_Forecast'] * cost_factor
    future_df['Adj_Cost'] = future_df['Cost_Forecast'] * cost_factor

    df['Adj_Cash_Flow'] = df['Adj_Revenue'] - df['Adj_Cost']
    future_df['Adj_Cash_Flow'] = future_df['Adj_Revenue'] - future_df['Adj_Cost']

    # --------- CHARTS ----------
    st.subheader("📈 Revenue Forecast")

    fig, ax = plt.subplots()
    ax.plot(df['Month'], df['Revenue'], label="Actual")
    ax.plot(df['Month'], df['Revenue_Forecast'], "--", label="Base Forecast")
    ax.plot(df['Month'], df['Adj_Revenue'], label="Adjusted Forecast", linewidth=3)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

    st.subheader("💰 Future Cash Flow Impact")

    fig2, ax2 = plt.subplots()
    ax2.bar(future_df['Month'], future_df['Adj_Cash_Flow'])
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # --------- IMPACT TABLE ----------
    st.subheader("📊 Scenario Impact Summary")

    impact_df = pd.DataFrame({
        "Metric": ["Revenue", "Total Cost", "Cash Flow"],
        "Base Forecast": [
            future_df['Revenue_Forecast'].sum(),
            future_df['Cost_Forecast'].sum(),
            (future_df['Revenue_Forecast'] - future_df['Cost_Forecast']).sum()
        ],
        "Adjusted Forecast": [
            future_df['Adj_Revenue'].sum(),
            future_df['Adj_Cost'].sum(),
            future_df['Adj_Cash_Flow'].sum()
        ]
    })

    impact_df["Variance"] = impact_df["Adjusted Forecast"] - impact_df["Base Forecast"]

    numeric_cols = ["Base Forecast", "Adjusted Forecast", "Variance"]
    st.dataframe(
        impact_df.style.format({col: "{:,.0f}" for col in numeric_cols})
    )

    # --------- DOWNLOAD ----------
    st.subheader("⬇ Download Forecast")

    download_df = pd.concat([df, future_df], ignore_index=True)
    csv = download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Forecast CSV",
        csv,
        "AI_Financial_Forecast_Report.csv",
        "text/csv"
    )

    # --------- ASSUMPTIONS ----------
    st.subheader("📘 Assumptions & Drivers")

    with st.expander("Forecasting Assumptions"):
        st.write("""
        • Linear trend continuation  
        • Stable seasonal pattern  
        • No extraordinary shocks  
        • Short-term planning horizon
        """)

    with st.expander("Key Drivers"):
        st.write("""
        • Time-based growth  
        • Month-wise seasonality  
        • Revenue & cost sensitivity  
        • Scenario-based management inputs
        """)

else:
    st.warning("👈 Please upload a CSV file to begin.")
