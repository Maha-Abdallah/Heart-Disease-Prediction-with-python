# Heart-Disease-Prediction-with-python

This project aims to predict heart disease by analyzing various health metrics. We use a dataset that includes features such as age, sex, blood pressure, and levels of various substances in the blood.

# Dataset

The dataset includes the following features:

Age: The person’s age in years
Sex: The person’s sex (1 = male, 0 = female)
Creatinine Phosphokinase: Level of the CPK enzyme in the blood
Diabetes: Whether the person has diabetes (1 = yes, 0 = no)
Ejection Fraction: Percentage of blood leaving the heart at each contraction
High Blood Pressure: Whether the person has hypertension (1 = yes, 0 = no)
Platelets: Platelets in the blood (kiloplatelets/mL)
Serum Creatinine: Level of serum creatinine in the blood (mg/dL)
Serum Sodium: Level of serum sodium in the blood (mEq/L)
Anaemia: Decrease of red blood cells or hemoglobin (1 = yes, 0 = no)

# The target variable is:

DEATH_EVENT: Whether the person had a heart disease event (1 = yes, 0 = no)

# Analysis

We performed an exploratory data analysis to understand the distribution of each feature and their correlation with each other. Here are some key findings:

The age distribution is skewed to the right, indicating that there are more people in the younger age groups than in the older age groups.
The most frequent age group is 40-50 years old.
There is a fair amount of variability in the age data, with the interquartile range spanning from 40 to 70 years old.
There is no clear linear relationship between creatinine phosphokinase and ejection fraction, suggesting that some people in the study had heart damage.
Age has a weak positive correlation with serum creatinine and a weak negative correlation with sex (female).
Anaemia has a weak positive correlation with sex (female) and high blood pressure.
Creatinine phosphokinase has a weak negative correlation with diabetes and ejection fraction.
Diabetes has a weak negative correlation with ejection fraction and sex (female).
High blood pressure has a weak negative correlation with ejection fraction.

# Model

We trained a Logistic Regression model and a Random Forest model to predict heart disease events. The Logistic Regression model achieved an accuracy of 80.00%, and the Random Forest model achieved an accuracy of 75.00%.

Please note that these models are initial models and we plan to try out more complex models and feature engineering techniques to improve the performance.

# Future Work

We plan to further explore the data and try out different machine learning models and techniques to improve the prediction accuracy. We also plan to investigate the outliers and understand their impact on the model’s performance.

Contributions and suggestions are welcome! 😊
