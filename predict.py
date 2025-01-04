import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "Prediction 1.xlsx"  # Update with your file path
df = pd.read_excel(file_path)

# Ensure the year column is formatted correctly and set as the index
df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
df = df[df['year'] > 0]  # Remove rows with invalid years
df.set_index('year', inplace=True)

# Aggregate Y1, Y2, Y3 by year
time_series = df[['Y1', 'Y2', 'Y3']].groupby(df.index).mean()


# Function to train SARIMA and forecast
def forecast_sarima(series, end_year=2030, seasonal_order=(1, 1, 1, 12), order=(1, 1, 1)):
    # Train SARIMA model
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Forecast future values
    forecast_years = range(series.index[-1] + 1, end_year + 1)
    forecast = model_fit.forecast(steps=len(forecast_years))

    # Combine actual and forecasted values
    forecast_series = pd.Series(forecast, index=forecast_years)
    return forecast_series, model_fit

# Forecast Y1, Y2, Y3 till 2030
forecasts = {}
seasonal_order = (1, 1, 1, 12)  # Seasonal parameters (adjust as needed)
for col in ['Y1', 'Y2', 'Y3']:
    forecasts[col], _ = forecast_sarima(time_series[col], seasonal_order=seasonal_order)

# Plot the results
plt.figure(figsize=(12, 8))
for col in ['Y1', 'Y2', 'Y3']:
    plt.plot(time_series[col], label=f'Actual {col}')
    plt.plot(forecasts[col], label=f'Forecast {col}', linestyle='--')
plt.title('SARIMA Forecasts for Y1, Y2, Y3')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()

# Save forecast results to Excel
forecast_df = pd.concat(forecasts, axis=1)
forecast_df.to_excel("SARIMA_Forecast_Results.xlsx")
print("Forecasts saved to 'SARIMA_Forecast_Results.xlsx'.")
