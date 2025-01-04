# pip install pandas matplotlib seaborn openpyxl
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
#pip install git+https://github.com/statsmodels/statsmodels.git

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Prediction 1.xlsx"  
data = pd.read_excel(file_path)

# Display the first few rows of the dataset
print(data.head())

data.dropna(inplace=True)  # Drop rows with missing values (if applicable)
data['year'] = data['year'].astype(int)  # Ensure 'year' is an integer


# 1. Line Plot: Trends of y1, y2, y3 over the years for a specific country
country = "indonesia"  # Specify the country
filtered_data = data[data['countryname'] == country]
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['year'], filtered_data['Y1'], label='Y1', marker='o')
plt.plot(filtered_data['year'], filtered_data['Y2'], label='Y2', marker='o')
plt.plot(filtered_data['year'], filtered_data['Y3'], label='Y3', marker='o')
plt.title(f'Trends of Y1, Y2, Y3 in {country}')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()

# 2. Bar Plot: Average y1, y2, y3 for each country
avg_values = data.groupby('countryname')[['Y1', 'Y2', 'Y3']].mean()
avg_values.plot(kind='bar', figsize=(12, 6))
plt.title('Average Y1, Y2, Y3 by Country')
plt.xlabel('Country')
plt.ylabel('Average Values')
plt.xticks(rotation=45)
plt.show()

# 3. Heatmap: Correlation between numeric columns
plt.figure(figsize=(8, 6))
sns.heatmap(data[['Y1', 'Y2', 'Y3']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 4. Scatter Plot: Relationship between y1 and y3
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Y1', y='Y3', hue='countryname', data=data)
plt.title('Y1 vs Y3')
plt.xlabel('Y1')
plt.ylabel('Y3')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()