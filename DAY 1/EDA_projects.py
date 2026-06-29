import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os 



print("Understanding the Dataset")


file_name = "sales_data.csv"

if not os.path.exists(file_name):
    
    print(f"Error : {file_name} is not found")
    exit()


df = pd.read_csv(file_name)
print("sucessfully loaded")
print(f"shape of the dataset:Rows:{df.shape[0]},columns:{df.shape[1]}")


print(df.head())
print(df.tail())
print(df.describe())

print("Handling Missing values")

print(df.isnull().sum())

median_age=df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(median_age)

mean_spending = df['Spending'].mean()
df['Spending'] = df['Spending'].fillna(mean_spending)
print(mean_spending)

plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10,color='red',edgecolor='black')
plt.title('Distribution of spending')
plt.xlabel('spending amount')
plt.ylabel('number of customers')
plt.show()


correlation = df.corr(numeric_only=True)
print(correlation)
print("Plotting Correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt='0.2f')
plt.title("Correlation Heatmap")
plt.show()

print("find the Outliers in age")
outliers = df[df['Age']>=100]
print("Found outliers(s):")
print(outliers)

print("Removing Outliers")
df_cleaned = df[df['Age']<100]

print("Shape of cleaned dataset")