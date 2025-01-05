# pip install pandas matplotlib seaborn openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Prediction Updated file.xlsx"  
data = pd.read_excel(file_path)

# Display the first few rows of the dataset
#print(data.head())

data.dropna(inplace=True)  # Drop rows with missing values (if applicable)
data['year'] = data['year'].astype(int)  # Ensure 'year' is an integer


# 1. Line Plot: Trends of y1, y2, y3 over the years for a specific country
country = "Thailand"  # Specify the country
filtered_data = data[data['countryname'] == country]
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['year'], filtered_data['Scope 1'], label='Scope 1', marker='o')
plt.plot(filtered_data['year'], filtered_data['Scope 2'], label='Scope 2', marker='o')
plt.plot(filtered_data['year'], filtered_data['Scope 3'], label='Scope 3', marker='o')
plt.title(f'Trends of Scope 1, Scope 2, Scope 3 in {country}')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()

# 2. Bar Plot: Average y1, y2, y3 for each country
avg_values = data.groupby('countryname')[['Scope 1', 'Scope 2', 'Scope 3']].mean()
avg_values.plot(kind='bar', figsize=(12, 6))
plt.title('Intensity Scope 1, Scope 2, Scope 3 by Country')
plt.xlabel('Country')
plt.ylabel('Intensity')
plt.xticks(rotation=45)
plt.show()

# 3. Heatmap: Correlation between numeric columns
plt.figure(figsize=(8, 6))
sns.heatmap(data[['Scope 1', 'Scope 2', 'Scope 3']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 4. Scatter Plot: Relationship between y1 and y3
plt.figure(figsize=(4, 6))
sns.scatterplot(x='Scope 2', y='Scope 3', hue='countryname', data=data)
plt.title('Scope 2 vs Scope 3')
plt.xlabel('Scope 2')
plt.ylabel('Scope 3')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()