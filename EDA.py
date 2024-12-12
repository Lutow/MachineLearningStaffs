import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "Credit Risk Dataset.xlsx"  # Path to your uploaded file
df = pd.read_excel(file_path)

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
pd.set_option('display.max_columns', None)
