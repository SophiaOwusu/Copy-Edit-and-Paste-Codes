# import numpy and pandas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# convert target column to object data type
df['target'] = df['target'].astype('object')

# convert date features to datetime data type
for column in ['Policy Start Date', 'Policy End Date', 'First Transaction Date']:
    df[column] = pd.to_datetime(df[column])
display(df.info())

# Renaming columns to use a uniform format
df = df.rename(columns={'Subject_Car_Make': 'Car_Make', 'Subject_Car_Colour': 'Car_Colour', 'Policy Start Date': 'Policy_Start_Date', 'Policy End Date': 'Policy_End_Date', 'First Transaction Date': 'First_Transaction_Date', 'No_Pol': 'Policy_Count', 'ProductName': 'Product_Name', 'target': 'Target'})
df.sample(2)

# EDA Data Distributions
sns.pairplot(df)
plt.show()
display(df.info())

# Categorical Features Distribution,List of columns to plot
cat_columns = ['ID', 'Gender', 'Car_Category', 'Car_Colour', 'Car_Make', 'LGA_Name', 'State', 'Product_Name', 'Target']
plt.figure(figsize=(18, 12))

# Iterate through the columns and create subplots
for i, column in enumerate(cat_columns, 1):
    plt.subplot(3, 3, i)
    ax = sns.countplot(x=df[column], data=df, edgecolor='black')
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.yticks([])
    # add data labels
    ax.bar_label(ax.containers[0], label_type='edge', fontsize=10)
plt.tight_layout()
plt.show()

# Numerical Features Distribution List of columns to plot
num_columns = ['Age', 'Policy_Count']

# Create the figure
plt.figure(figsize=(14, 8))

# Iterate through the columns and create subplots
for c, column in enumerate(num_columns, 1):
    plt.subplot(2, 3, c)
    sns.histplot(df[column], kde=True, bins = 20)
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Check outliers
sns.boxplot(x=df['Age'], color='#0F3D15')
plt.show()

# check correlation using Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, color='#0F3D15')
plt.show()

# extract all categorical columns
categorical_columns =df.select_dtypes(include = 'object').columns.tolist()

# for loop to change categorical columns with object data types
for column in categorical_columns:
    df[column] = df[column].astype('category')
display(df.info())

# Handling missing values in the Gender, Car_Category, Car_Colour, Car_Make, LGA_Name, and State Columns
columns_to_fill = ['Gender', 'Car_Category', 'Car_Colour', 'Car_Make', 'LGA_Name', 'State']

# filling each colomn with respective mode due to outliers restricting from using mean
for column in columns_to_fill:
    df[column] = df[column].fillna(df[column].mode()[0])
df.isnull().sum()

# Handling outliers in the age column
'''
Male, Female, Joint = Ages >= 18 and <= 75
Entity, Unkown = Ages >= 1 and <= 116
'''
# Define the condition for valid ages
condition = ((df['Gender'].isin(['Male', 'Female', 'Joint Gender']) & df['Age'].between(18, 75, inclusive='both')) | (df['Gender'].isin(['Entity', 'Unknown']) & df['Age'].between(1, 116, inclusive='both')))
# Calculate the total number of entries
total_count = len(df)
# Identify the outliers (rows that *don't* meet the condition)
outliers = ~condition
outlier_count = outliers.sum() # sum of True values gives the count
# Calculate the percentage of outliers
outlier_percentage = (outlier_count / total_count) * 100
print(f"Total entries: {total_count}")
print(f"Outlier entries (ages that don't meet the condition) out of the {total_count}: {outlier_count} ({outlier_percentage:.0f}%)")

# Calculate the median of the 'Age' column
median_age = round(df['Age'].median())
print("The median of the age column is:", median_age)

# replacing outliers (entries that do not meet the condition) with the median
'''Replace ages that don't meet the condition with the median age'''
df.loc[~condition, 'Age'] = 40
df.sample(10)
# confirm if outliers have been handled with histplot
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, color='#0F3D15')
plt.show()

# Checking column distribution with pie chart
gender_counts = df['Gender'].dropna().value_counts()
custom_colors = ['#1d3557', '#457b9d', '#a8dadc', '#f1faee', '#e63946']
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=None, colors=custom_colors[:len(gender_counts)],autopct='%1.1f%%', startangle=90)
plt.legend(gender_counts.index, title='Gender', loc='best', bbox_to_anchor=(1, 0.5))
plt.title('Gender Distribution')
plt.show()

# check distribution with barchart(subplots) and data labels
product_counts = df['Product_Name'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
bars = product_counts.sort_values().plot(kind='barh', ax=ax, color='#0F3D15')
ax.set_title('Distribution of Products')
ax.set_ylabel('Product Name')
ax.set_xticks([])
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=10)
plt.show()

# check distribution with columnchart(barplots) and data labels
target_counts = df['Target'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=target_counts.index, y=target_counts.values, palette='viridis')
plt.title('Distribution of Target Variable (Claim Within 3 Months)')
plt.xlabel('Target (0 = No Claim, 1 = Claim)')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Claim', 'Claim'])
plt.tight_layout()
plt.show()
# target proportions
print("Target Proportions:")
print(target_counts / len(df))

# check for duplicates
df[df.duplicated()]
# remove the duplicates
df = df.drop_duplicates()
# check for duplicates again
display(df[df.duplicated()])
# check the new shape
display(df.shape)

# Exporting file(csv)
df.to_csv('cleaned_insurance_company.csv', index=False)

# Feature Engineering
#Dropping 'ID', 'Policy_Start_Date', 'Policy_End_Date','First_Transaction_Date', 'Car_Colour'
'''
As these columns are not necessary for our modeling
'''
df = df.drop(['ID', 'Policy_Start_Date', 'Policy_End_Date', 'First_Transaction_Date', 'Car_Colour'], axis=1)
display(df.head(2))
display(df.info())

