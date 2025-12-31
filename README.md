# FinSight-AI
FinSight AI is an AI-assisted financial forecasting web application designed to support corporate budgeting, forecasting, and decision-making. The model uses historical financial data to generate future projections of revenue, costs, and cash flows using trend analysis with optional seasonality. It also enables scenario analysis through adjustable business drivers.

The application is built using Python and Streamlit, making it interactive, explainable, and easy to use for non-technical users.

🎯 Key Features

📈 Revenue, cost, and cash flow forecasting

🔄 Trend-based forecasting with optional seasonality

🧠 Scenario analysis: Basic, Optimistic, Pessimistic

🎛 Adjustable business drivers to simulate management decisions

📊 Interactive charts and summary tables

🌐 Simple, browser-based Streamlit interface

🏢 Corporate Use Cases

Financial Planning & Analysis (FP&A)

Budgeting and forecasting

Cash flow planning

Risk and scenario analysis

Management decision support

📁 Input Data Format

The application accepts a CSV file with the following columns:

Month, Revenue, Fixed_Cost, Variable_Cost


Month should be in YYYY-MM-DD format

Values should be numeric

Sample CSV files are included in the repository.

🛠 Technology Stack

Python

Streamlit – Web application interface

Pandas – Data handling

NumPy – Numerical computations

Matplotlib – Visualizations

Scikit-learn – Trend modeling

▶ How to Run the App Locally
1️⃣ Install dependencies
pip install streamlit pandas numpy matplotlib scikit-learn

2️⃣ Run the application
streamlit run ai_assisted_forecasting_model.py


The app will open automatically in your web browser.

🌐 Live Demo (If Deployed)

If deployed on Streamlit Community Cloud, access the live app here:
👉 (Add your Streamlit URL here)

📌 Assumptions

Historical trends continue into the forecast period

Seasonal patterns repeat consistently

Business drivers remain constant during the forecast horizon

No major economic or structural shocks occur

⚠ Limitations

Not designed for high-frequency or real-time forecasting

Does not capture sudden market disruptions

Accuracy depends on quality of historical data

📚 Academic Context

This project was developed as part of an AI in Finance / Corporate Analytics initiative and is suitable for academic evaluation, demonstrations, and learning purposes.

📜 License

This project is licensed under the MIT License — free to use, modify, and distribute with attribution.

🙋 Author

Niranjani
AI in Finance Project
ICAI AI Hub

⭐ Final Note

FinSight AI demonstrates how AI-assisted forecasting, combined with managerial judgment, can provide practical and explainable insights for corporate financial decision-making.
