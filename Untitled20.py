#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Load the dataset
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')


# In[3]:


# Data cleaning (if necessary)
# In this case, let's assume the data is already clean

# Basic statistics
print(df.describe())


# # Data visualization

# In[4]:


# Histogram of age
plt.hist(df['age'], bins=10, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age')
plt.show()


# In[ ]:


# Histogram of Age: This plot provides a visual representation of the distribution of ages in the dataset. The x-axis represents the age and the y-axis represents the count of individuals at each age. The bars represent different age intervals, showing how many individuals fall into each age range.
# The distribution of ages is skewed to the right. This means that there are more people in the younger age groups than in the older age groups.
# The most frequent age group is 40-50 years old.
# There are very few people in the 90 or older age group.


# In[5]:


# Boxplot for age
sns.boxplot(y='age', data=df)
plt.title('Boxplot of Age')
plt.show()


# In[ ]:


# This plot provides a summary of the age variable. The box represents the interquartile range (IQR), the line inside the box is the median, and the whiskers represent the range of the data within 1.5 times the IQR.
# The median age is around 55 years old.
# There is a fair amount of variability in the data, with the IQR spanning from 40 to 70 years old.
# There are a few outliers on the higher end, with some people being 90 years old or older.


# In[6]:


# Scatter plot for creatinine_phosphokinase and ejection_fraction
plt.scatter(df['creatinine_phosphokinase'], df['ejection_fraction'])
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Ejection Fraction')
plt.title('Scatter plot of Creatinine Phosphokinase vs Ejection Fraction')
plt.show()


# In[ ]:


# This plot shows the relationship between the two variables. Each point represents an observation from the dataset and its position on the x and y axes indicates its values for the two variables.
# There is no clear linear relationship between creatinine phosphokinase and ejection fraction.
# There is a cloud of points spread throughout the plot, with no strong upward or downward trend.
# Some data points have high values for creatinine phosphokinase and low values for ejection fraction. This could indicate that some people in the study had heart damage.


# In[7]:


# Correlation matrix heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[ ]:


# This plot provides a visual representation of the correlation between every pair of variables in the dataset. The color and intensity of each cell in the grid represent the strength and direction (positive/negative) of the correlation.
# Age has a weak positive correlation with serum creatinine and a weak negative correlation with sex (female). This means that as age increases, serum creatinine tends to increase slightly, and men are more likely to have higher serum creatinine levels than women.
# Anaemia has a weak positive correlation with sex (female) and high blood pressure. This means that women are slightly more likely to have anaemia than men, and there is a slight positive correlation between anaemia and high blood pressure.
# Creatinine phosphokinase has a weak negative correlation with diabetes and ejection fraction. This means there is a slight negative correlation between creatine phosphokinase levels and both diabetes and ejection fraction.
# Diabetes has a weak negative correlation with ejection fraction and sex (female). This means there is a slight negative correlation between diabetes and both ejection fraction and being female.
# High blood pressure has a weak negative correlation with ejection fraction. This means there is a slight negative correlation between high blood pressure and ejection fraction.


# # Machine Learning

# In[8]:


# Split the data into features (X) and target (y)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']


# In[9]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[11]:


# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[12]:


# Make predictions on the test set and calculate the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy*100:.2f}%')


# In[ ]:


# An accuracy of 80.00% means that your model correctly predicted the outcome in 80% of cases in the test set. This is a fairly good accuracy score, especially for a simple model like Logistic Regression, and suggests that your model has learned to capture the underlying patterns in your data quite well.


# In[13]:


# Train a Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)


# In[14]:


# Make predictions on the test set and calculate the accuracy
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf*100:.2f}%')


# In[ ]:


# An accuracy of 75.00% means that your model correctly predicted the outcome in 75% of cases in the test set. This is a fairly good accuracy score, especially for a complex model like Random Forest, and suggests that your model has learned to capture the underlying patterns in your data quite well.

