import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

def plot_arima_forecast(df, model_name, ax):
    try:
        forecast_steps = 15  # Define forecast_steps
        df = df.groupby('Date').sum()
        if df.empty:
            ax.set_title(f'{model_name} (No data)')
            return
        model = ARIMA(df['Daily_Sales_Quantity'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        ax.plot(df.index, df['Daily_Sales_Quantity'], label='Original')
        forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        ax.plot(forecast_index, forecast, label='Forecast')
        ax.set_title(model_name)
        ax.legend()
    except Exception as e:
        ax.set_title(f'{model_name} (Error)')
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

def plot_sales(ax, model_name, df, color):
    model_df = df[df['Model'] == model_name]
    if model_df.empty:
        st.write(f"No data available for {model_name}")
        ax.set_title(f'{model_name} (No Data)')
        return
    model_df = model_df.groupby('Date').sum()
    ax.plot(model_df.index, model_df['Daily_Sales_Quantity'], color=color)
    ax.set_title(model_name)

# Streamlit app
def main():
    st.title("Demand Forecasting and Visualization")

    # Initialize variables
    df = None

    # File uploader for univariate dataset
    uploaded_file_uni = st.file_uploader("Upload Univariate Dataset", type="xlsx")
    if uploaded_file_uni:
        df = pd.read_excel(uploaded_file_uni)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Handle date parsing errors

        # Check if 'Date' column and 'Daily_Sales_Quantity' are correctly parsed
        if df['Date'].isnull().any() or df['Daily_Sales_Quantity'].isnull().any():
            st.write("Warning: There are missing values in the 'Date' or 'Daily_Sales_Quantity' columns.")

        # Plot historical sales data
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        if df is not None:
            plot_sales(axs[0, 0], 'Backhoe Loader', df, 'blue')
            plot_sales(axs[0, 1], 'Excavators(crawler)', df, 'green')
            plot_sales(axs[1, 0], 'Loaders (Wheeled)', df, 'red')
            plot_sales(axs[1, 1], 'Skid Steer Loaders', df, 'purple')
            plot_sales(axs[2, 0], 'Compactors', df, 'orange')
            plot_sales(axs[2, 1], 'Tele_Handlers', df, 'brown')

            plt.tight_layout()
            st.pyplot(fig)

    # File uploader for multivariate dataset
    uploaded_file_multi = st.file_uploader("Upload Multivariate Dataset", type="xlsx")
    if uploaded_file_multi:
        data = pd.read_excel(uploaded_file_multi)

        # Process the multivariate dataset
        C = data[["Value", "Month", "Monthpercent", "Day", "percent", "Market_Share", "political", "Marketing", "Budget"]]
        X = data[["Value", "Month", "Monthpercent", "Day", "percent", "Market_Share", "political", "Marketing", "Budget"]]
        y = data["Daily_Sales_Quantity"]

        # Ensure all features are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        C = C.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values if any
        X = X.dropna()
        C = C.dropna()
        y = y[X.index]  # Align y with X after dropping NaN

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Random Forest model
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)

        # Predict and add results to DataFrame
        y_pred = rf_regressor.predict(C)
        result = pd.DataFrame(y_pred, columns=['Sales'])
        C = C.copy()  # Ensure original C is not overwritten
        C["Sales"] = result['Sales'].round()
        st.write(C)

        # Forecasting with ARIMA
        if df is not None:
            forecast_steps = 15
            fig, axs = plt.subplots(3, 2, figsize=(15, 15))

            # Plot ARIMA forecasts
            plot_arima_forecast(df[df['Model'] == 'Backhoe Loader'], 'Backhoe Loader', axs[0, 0])
            plot_arima_forecast(df[df['Model'] == 'Excavators(crawler)'], 'Excavators', axs[0, 1])
            plot_arima_forecast(df[df['Model'] == 'Loaders (Wheeled)'], 'Loaders', axs[1, 0])
            plot_arima_forecast(df[df['Model'] == 'Skid Steer Loaders'], 'Skid Steer Loaders', axs[1, 1])
            plot_arima_forecast(df[df['Model'] == 'Compactors'], 'Compactors', axs[2, 0])
            plot_arima_forecast(df[df['Model'] == 'Tele_Handlers'], 'Tele_Handlers', axs[2, 1])

            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
