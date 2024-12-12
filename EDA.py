import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "Credit Risk Dataset.xlsx"  # Path to your uploaded file
df = pd.read_excel(file_path)

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
pd.set_option('display.max_columns', None)

summary = pd.DataFrame({
    'Mean': df[numerical_columns].mean(),
    'Median': df[numerical_columns].median(),
    'Variance': df[numerical_columns].var(),
    'Standard Deviation': df[numerical_columns].std(),
})

# Select numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

print(summary)
