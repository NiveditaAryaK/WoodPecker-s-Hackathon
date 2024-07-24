import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

st.title('Demand Forecasting and ABC Analysis App')

st.sidebar.header('Upload Files')

univariant_file = st.sidebar.file_uploader("Upload Univariant Data", type="xlsx")
multivariant_file = st.sidebar.file_uploader("Upload Multivariant Data", type="xlsx")
raw_material_file = st.sidebar.file_uploader("Upload Raw Material Data", type="xlsx")
input_file = st.sidebar.file_uploader("Upload Input Data", type="xlsx")

if univariant_file and multivariant_file and raw_material_file and input_file:
    # Load Data
    df = pd.read_excel(univariant_file)
    data = pd.read_excel(multivariant_file)
    raw_material_backhoe_loader = pd.read_excel(raw_material_file, sheet_name=0)
    raw_material_excavators = pd.read_excel(raw_material_file, sheet_name=1)
    raw_material_loaders = pd.read_excel(raw_material_file, sheet_name=2)
    raw_material_skid_steer_loaders = pd.read_excel(raw_material_file, sheet_name=3)
    raw_material_compactors = pd.read_excel(raw_material_file, sheet_name=4)
    raw_material_tele_handlers = pd.read_excel(raw_material_file, sheet_name=5)
    jp = pd.read_excel(input_file)

    # Data Preparation
    C = jp[["Value","Month","Monthpercent","Day","percent","Market_Share","political","Marketing","Budget"]]
    X = data[["Value","Month","Monthpercent","Day","percent","Market_Share","political","Marketing","Budget"]]
    y = data["Daily_Sales_Quantity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(C)
    result = pd.DataFrame(y_pred, columns=['Sales'])
    C["Sales"] = result["Sales"].round()

    # Forecasting
    univariant_data = []

    def plot_forecast(ax, df, model_name):
        if df.empty:
            st.warning(f"No data available for {model_name}. Skipping forecast.")
            return pd.DataFrame()  # Return an empty DataFrame if no data

        df = df.groupby('Date').sum()
        model = ARIMA(df['Daily_Sales_Quantity'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast_steps = int(len(jp)/6)
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        ax.plot(df.index, df['Daily_Sales_Quantity'], label='Original')
        ax.plot(forecast_dates, forecast, label='Forecast')
        ax.set_title(model_name)
        ax.legend()
        return pd.DataFrame({'Date': forecast_dates, 'Model': model_name, 'Forecast': forecast})

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    univariant_data.append(plot_forecast(axs[0, 0], df[df['Model'] == 'Backhoe Loader'], 'Backhoe Loader'))
    univariant_data.append(plot_forecast(axs[0, 1], df[df['Model'] == 'Excavators(crawler)'], 'Excavators'))
    univariant_data.append(plot_forecast(axs[1, 0], df[df['Model'] == 'Loaders (Wheeled)'], 'Loaders'))
    univariant_data.append(plot_forecast(axs[1, 1], df[df['Model'] == 'Skid Steer Loaders'], 'Skid Steer Loaders'))
    univariant_data.append(plot_forecast(axs[2, 0], df[df['Model'] == 'Compactors'], 'Compactors'))
    univariant_data.append(plot_forecast(axs[2, 1], df[df['Model'] == 'Tele_Handlers '], 'Tele Handlers'))

    st.pyplot(fig)

    # Define raw_material_dict
    raw_material_dict = {
        'Backhoe Loader': raw_material_backhoe_loader,
        'Excavators (crawler)': raw_material_excavators,
        'Loaders (Wheeled)': raw_material_loaders,
        'Skid Steer Loaders': raw_material_skid_steer_loaders,
        'Compactors': raw_material_compactors,
        'Tele Handlers': raw_material_tele_handlers
    }

    def perform_abc_analysis(demand_df, raw_material_dict):
        results = []
        for index, row in demand_df.iterrows():
            model = row['Model']
            forecast = row['Forecast']

            if model in raw_material_dict:
                raw_material_data = raw_material_dict[model]

                # Calculate cost contribution based on demand
                raw_material_data['Total Cost Based on Demand'] = raw_material_data['total'] * forecast

                # Sort by cost
                raw_material_sorted = raw_material_data.sort_values(by='Total Cost Based on Demand', ascending=False)

                # Calculate cumulative total and percentage
                raw_material_sorted['Cumulative Total'] = raw_material_sorted['Total Cost Based on Demand'].cumsum()
                raw_material_sorted['Cumulative Percentage'] = raw_material_sorted['Cumulative Total'] / raw_material_sorted['Total Cost Based on Demand'].sum()

                # Assign ABC categories
                def assign_abc(row):
                    if row['Cumulative Percentage'] <= 0.8:
                        return 'A'
                    elif row['Cumulative Percentage'] <= 0.95:
                        return 'B'
                    else:
                        return 'C'

                raw_material_sorted['ABC'] = raw_material_sorted.apply(assign_abc, axis=1)

                # Add machine model to results
                raw_material_sorted['Model'] = model
                results.append(raw_material_sorted)

        # Concatenate all results into a single DataFrame
        all_results_df = pd.concat(results, ignore_index=True)
        return all_results_df

    # Perform ABC analysis and display results for univariant data
    abc_results_df = perform_abc_analysis(pd.concat(univariant_data, ignore_index=True), raw_material_dict)
    
    # Filter for category A
    abc_results_df = abc_results_df[abc_results_df['ABC'] == 'A']

    grouped_df = abc_results_df.groupby('Name').agg({
        'quantity': 'sum',
        'price': 'mean',
        'total': 'sum',
        'Total Cost Based on Demand': 'sum',
        'Cumulative Total': 'max',
        'Cumulative Percentage': 'max',
        'ABC': 'first'
    }).reset_index()

    st.write("### Univariant Forecast and ABC Analysis")
    st.write(grouped_df)

    # Multivariant Forecasting
    def plot_multivariant_forecast(ax, df, model_name):
        if df.empty:
            return pd.DataFrame()  # Return an empty DataFrame if no data

        df = df.groupby('Date').sum()
        model = ARIMA(df['Daily_Sales_Quantity'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast_steps = int(len(jp)/6)
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        ax.plot(df.index, df['Daily_Sales_Quantity'], label='Original')
        ax.plot(forecast_dates, forecast, label='Forecast')
        ax.set_title(model_name)
        ax.legend()
        return pd.DataFrame({'Date': forecast_dates, 'Model': model_name, 'Forecast': forecast})

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    multivariant_data = []
    multivariant_data.append(plot_multivariant_forecast(axs[0, 0], data[data['Model'] == 'Backhoe Loader'], 'Backhoe Loader'))
    multivariant_data.append(plot_multivariant_forecast(axs[0, 1], data[data['Model'] == 'Excavators(crawler)'], 'Excavators'))
    multivariant_data.append(plot_multivariant_forecast(axs[1, 0], data[data['Model'] == 'Loaders (Wheeled)'], 'Loaders'))
    multivariant_data.append(plot_multivariant_forecast(axs[1, 1], data[data['Model'] == 'Skid Steer Loaders'], 'Skid Steer Loaders'))
    multivariant_data.append(plot_multivariant_forecast(axs[2, 0], data[data['Model'] == 'Compactors'], 'Compactors'))
    multivariant_data.append(plot_multivariant_forecast(axs[2, 1], data[data['Model'] == 'Tele_Handlers '], 'Tele Handlers'))

    # Define raw_material_dict for multivariant data
    raw_material_dict1 = {
        'Backhoe Loader': raw_material_backhoe_loader,
        'Excavators(crawler)': raw_material_excavators,
        'Loaders (Wheeled)': raw_material_loaders,
        'Skid Steer Loaders': raw_material_skid_steer_loaders,
        'Compactors': raw_material_compactors,
        'Tele_Handlers ': raw_material_tele_handlers
    }

    # Perform ABC analysis and display results for multivariant data
    abc_results_df1 = perform_abc_analysis(pd.concat(multivariant_data, ignore_index=True), raw_material_dict1)
    
    # Filter for categories B and C
    abc_results_df1 = abc_results_df1[abc_results_df1['ABC'].isin(['B', 'C'])]

    grouped_df1 = abc_results_df1.groupby('Name').agg({
        'quantity': 'sum',
        'price': 'mean',
        'total': 'sum',
        'Total Cost Based on Demand': 'sum',
        'Cumulative Total': 'max',
        'Cumulative Percentage': 'max',
        'ABC': 'first'
    }).reset_index()

    st.write("### Multivariant Forecast and ABC Analysis")
    st.write(grouped_df1)

else:
    st.write("Please upload all the required files.")