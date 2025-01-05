import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "Prediction Updated file.xlsx"  # Update with your file path
df = pd.read_excel(file_path)

# Preprocess data
df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
df = df[df['year'] > 0]
df.set_index('year', inplace=True)

# Convert the index to a PeriodIndex for better compatibility with SARIMAX
df.index = pd.PeriodIndex(df.index, freq='Y')

# Function to train SARIMA and forecast
def forecast_sarima(series, end_year=2030, seasonal_order=(1, 1, 1, 12), order=(1, 1, 1)):
    # Train SARIMA model
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Forecast future values
    forecast_periods = pd.period_range(start=series.index[-1] + 1, end=str(end_year), freq='Y')
    forecast = model_fit.forecast(steps=len(forecast_periods))

    # Combine actual and forecasted values
    forecast_series = pd.Series(forecast, index=forecast_periods)
    return forecast_series, model_fit

# Group data by country and create separate charts
countries = df['countryname'].unique()

for country in countries:
    # Filter data for the current country
    country_data = df[df['countryname'] == country]
    time_series = country_data[['Scope 1', 'Scope 2', 'Scope 3']].groupby(country_data.index).mean()

    # Forecast for each scope
    forecasts = {}
    seasonal_order = (1, 1, 1, 12)  # Seasonal parameters (adjust as needed)
    for col in ['Scope 1', 'Scope 2', 'Scope 3']:
        forecasts[col], _ = forecast_sarima(time_series[col], seasonal_order=seasonal_order)

    # Convert the PeriodIndex to DatetimeIndex for plotting
    time_series.index = time_series.index.to_timestamp()

    # Create a plot for the current country
    plt.figure(figsize=(12, 8))
    for col in ['Scope 1', 'Scope 2', 'Scope 3']:
        plt.plot(time_series.index, time_series[col], label=f'Observed {col}')
        plt.plot(forecasts[col].index.to_timestamp(), forecasts[col], label=f'Forecasted {col}')

    plt.title(f'SARIMA Forecasts for {country}')
    plt.xlabel('Year')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid()
    plt.show()

# Optional: Save forecast results to Excel for each country
all_forecasts = []
for country in countries:
    country_data = df[df['countryname'] == country]
    time_series = country_data[['Scope 1', 'Scope 2', 'Scope 3']].groupby(country_data.index).mean()

    forecasts = {}
    for col in ['Scope 1', 'Scope 2', 'Scope 3']:
        forecasts[col], _ = forecast_sarima(time_series[col], seasonal_order=seasonal_order)

    forecast_df = pd.concat(forecasts, axis=1)
    forecast_df.columns = [f'{country} - Forecast {col}' for col in ['Scope 1', 'Scope 2', 'Scope 3']]
    all_forecasts.append(forecast_df)

# Combine all forecasts into a single DataFrame and save to Excel
final_forecast_df = pd.concat(all_forecasts, axis=1)
final_forecast_df.to_excel("SARIMA_Forecast_Results_By_Country.xlsx")
print("Forecasts saved to 'SARIMA_Forecast_Results_By_Country.xlsx'.")
