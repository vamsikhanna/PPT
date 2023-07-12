#!/usr/bin/env python
# coding: utf-8

# 1. Scenario: A company wants to analyze the sales performance of its products in different regions. They have collected the following data:
#    Region A: [10, 15, 12, 8, 14]
#    Region B: [18, 20, 16, 22, 25]
#    Calculate the mean sales for each region.
# 

# In[ ]:


For Region A:
Sales data: [10, 15, 12, 8, 14]
Total sales for Region A: 10 + 15 + 12 + 8 + 14 = 59
Number of sales data points in Region A: 5

Mean sales for Region A: Total sales for Region A / Number of sales data points in Region A
Mean sales for Region A: 59 / 5 = 11.8

Therefore, the mean sales for Region A is 11.8.

For Region B:
Sales data: [18, 20, 16, 22, 25]
Total sales for Region B: 18 + 20 + 16 + 22 + 25 = 101
Number of sales data points in Region B: 5

Mean sales for Region B: Total sales for Region B / Number of sales data points in Region B
Mean sales for Region B: 101 / 5 = 20.2

Therefore, the mean sales for Region B is 20.2.








# 2. Scenario: A survey is conducted to measure customer satisfaction on a scale of 1 to 5. The data collected is as follows:
#    [4, 5, 2, 3, 5, 4, 3, 2, 4, 5]
#    Calculate the mode of the survey responses.
# 

# In[ ]:


To calculate the mode of the survey responses, you need to determine the value that appears most frequently in the data set. In this case, the survey responses are [4, 5, 2, 3, 5, 4, 3, 2, 4, 5].

To find the mode, we count the frequency of each unique value and identify the one with the highest frequency.

Frequency of 2: 2
Frequency of 3: 2
Frequency of 4: 3
Frequency of 5: 3

Since both 4 and 5 have the highest frequency of 3, the mode of the survey responses is 4 and 5.


# 3. Scenario: A company wants to compare the salaries of two departments. The salary data for Department A and Department B are as follows:
#    Department A: [5000, 6000, 5500, 7000]
#    Department B: [4500, 5500, 5800, 6000, 5200]
#    Calculate the median salary for each department.
# 

# In[ ]:


To calculate the median salary for each department, you need to arrange the salary data in ascending order and find the middle value. If the number of values is even, you take the average of the two middle values. Here's how you can calculate the median salary for Department A and Department B:

For Department A:
Salary data: [5000, 6000, 5500, 7000]
Arrange the data in ascending order: [5000, 5500, 6000, 7000]

Since the number of values is odd (4), the median is the middle value.
Median salary for Department A: 5500

For Department B:
Salary data: [4500, 5500, 5800, 6000, 5200]
Arrange the data in ascending order: [4500, 5200, 5500, 5800, 6000]

Since the number of values is odd (5), the median is the middle value.
Median salary for Department B: 5500

Therefore, the median salary for both Department A and Department B is 5500.








# 4. Scenario: A data analyst wants to determine the variability in the daily stock prices of a company. The data collected is as follows:
#    [25.5, 24.8, 26.1, 25.3, 24.9]
#    Calculate the range of the stock prices.
# 

# In[ ]:


To calculate the range of the stock prices, you need to find the difference between the highest and lowest values in the dataset. In this case, the stock prices are [25.5, 24.8, 26.1, 25.3, 24.9].

To find the range, follow these steps:

1. Find the highest value in the dataset: 26.1
2. Find the lowest value in the dataset: 24.8
3. Calculate the range by subtracting the lowest value from the highest value:
   Range = Highest value - Lowest value
   Range = 26.1 - 24.8
   Range = 1.3

Therefore, the range of the stock prices is 1.3.


# 5. Scenario: A study is conducted to compare the performance of two different teaching methods. The test scores of the students in each group are as follows:
#    Group A: [85, 90, 92, 88, 91]
#    Group B: [82, 88, 90, 86, 87]
#    Perform a t-test to determine if there is a significant difference in the mean scores between the two groups
# 

# In[ ]:


To perform a t-test and determine if there is a significant difference in the mean scores between Group A and Group B, you can follow these steps:

Step 1: State the null hypothesis (H0) and the alternative hypothesis (H1):
   - Null hypothesis (H0): There is no significant difference in the mean scores between Group A and Group B.
   - Alternative hypothesis (H1): There is a significant difference in the mean scores between Group A and Group B.

Step 2: Calculate the means of the two groups:
   Mean of Group A: (85 + 90 + 92 + 88 + 91) / 5 = 89.2
   Mean of Group B: (82 + 88 + 90 + 86 + 87) / 5 = 86.6

Step 3: Calculate the sample standard deviations of the two groups:
   For Group A:
     - Calculate the squared deviations from the mean: (85 - 89.2)^2, (90 - 89.2)^2, (92 - 89.2)^2, (88 - 89.2)^2, (91 - 89.2)^2
     - Sum the squared deviations: 16.96 + 0.64 + 5.76 + 1.44 + 1.44 = 26.24
     - Divide the sum by (n - 1) to calculate the sample variance: 26.24 / (5 - 1) = 6.56
     - Take the square root of the sample variance to obtain the sample standard deviation: √6.56 ≈ 2.56

   For Group B:
     - Calculate the squared deviations from the mean: (82 - 86.6)^2, (88 - 86.6)^2, (90 - 86.6)^2, (86 - 86.6)^2, (87 - 86.6)^2
     - Sum the squared deviations: 14.44 + 1.96 + 6.76 + 0.36 + 0.16 = 23.68
     - Divide the sum by (n - 1) to calculate the sample variance: 23.68 / (5 - 1) = 5.92
     - Take the square root of the sample variance to obtain the sample standard deviation: √5.92 ≈ 2.43

Step 4: Calculate the t-statistic:
   - The formula for the t-statistic is: t = (mean(Group A) - mean(Group B)) / √((s^2(Group A)/n(Group A)) + (s^2(Group B)/n(Group B)))
   - Substituting the values:
     t = (89.2 - 86.6) / √((2.56^2/5) + (2.43^2/5))
     t ≈ 2.6

Step 5: Determine the critical value and degrees of freedom:
   - The critical value depends on the desired significance level (e.g., 0.05) and the degrees of freedom, which is calculated as (n(Group A) + n(Group B)) - 2. In this case, the degrees of freedom is 8.
   - Using a t-table or statistical software, you can find the critical value for a two-tailed t-test with a significance level of 0.05 and 8 degrees of freedom. Let's assume the critical value is approximately 2.31.

Step 6: Compare the t-statistic with the critical value:
   - If the absolute value of the t-statistic is greater than the critical value, we reject the null hypothesis (H0) and conclude that there is a significant difference in the mean scores between the two groups.
   - In this case, |2.6| > 2.31, so we can reject the null hypothesis.

Therefore, based on the t-test, we can conclude that there is a significant difference in the mean scores between Group A and Group B.


# 6. Scenario: A company wants to analyze the relationship between advertising expenditure and sales. The data collected is as follows:
#    Advertising Expenditure (in thousands): [10, 15, 12, 8, 14]
#    Sales (in thousands): [25, 30, 28, 20, 26]
#    Calculate the correlation coefficient between advertising expenditure and sales.
# 
# 

# In[1]:


import numpy as np

# Create the data
advertising_expenditure = [10, 15, 12, 8, 14]
sales = [25, 30, 28, 20, 26]

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(advertising_expenditure, sales)[0, 1]

# Print the correlation coefficient
print(correlation_coefficient)


# 7. Scenario: A survey is conducted to measure the heights of a group of people. The data collected is as follows:
#    [160, 170, 165, 155, 175, 180, 170]
#    Calculate the standard deviation of the heights.
# 

# In[ ]:


import statistics

# Create the data
heights = [160, 170, 165, 155, 175, 180, 170]

# Calculate the mean
mean = statistics.mean(heights)

# Calculate the squared deviations from the mean
squared_deviations = []
for height in heights:
  deviation = height - mean
  squared_deviation = deviation ** 2
  squared_deviations.append(squared_deviation)

# Calculate the variance
variance = statistics.variance(heights)

# Calculate the standard deviation
standard_deviation = statistics.stdev(heights)

# Print the standard deviation
print(standard_deviation)


# 8. Scenario: A company wants to analyze the relationship between employee tenure and job satisfaction. The data collected is as follows:
#    Employee Tenure (in years): [2, 3, 5, 4, 6, 2, 4]
#    Job Satisfaction (on a scale of 1 to 10): [7, 8, 6, 9, 5, 7, 6]
#    Perform a linear regression analysis to predict job satisfaction based on employee tenure.
# 
# 

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create the data
tenure = np.array([2, 3, 5, 4, 6, 2, 4])
satisfaction = np.array([7, 8, 6, 9, 5, 7, 6])

# Create the linear regression model
model = LinearRegression()
model.fit(tenure, satisfaction)

# Print the coefficients
print(model.coef_)
print(model.intercept_)

# Plot the data and the regression line
plt.scatter(tenure, satisfaction)
plt.plot(tenure, model.predict(tenure))
plt.show()


# 9. Scenario: A study is conducted to compare the effectiveness of two different medications. The recovery times of the patients in each group are as follows:
#    Medication A: [10, 12, 14, 11, 13]
#    Medication B: [15, 17, 16, 14, 18]
#    Perform an analysis of variance (ANOVA) to determine if there is a significant difference in the mean recovery times between the two medications.
# 

# In[4]:


import numpy as np
import scipy.stats as stats

# Create the data
medication_a = np.array([10, 12, 14, 11, 13])
medication_b = np.array([15, 17, 16, 14, 18])

# Perform the ANOVA
f_statistic, p_value = stats.f_oneway(medication_a, medication_b)

# Print the results
print('F-statistic:', f_statistic)
print('p-value:', p_value)



# 10. Scenario: A company wants to analyze customer feedback ratings on a scale of 1 to 10. The data collected is
# 
#  as follows:
#     [8, 9, 7, 6, 8, 10, 9, 8, 7, 8]
#     Calculate the 75th percentile of the feedback ratings.
# 

# In[5]:


import statistics

# Create the data
feedback_ratings = [8, 9, 7, 6, 8, 10, 9, 8, 7, 8]

# Calculate the 75th percentile
percentile = 0.75
sorted_ratings = sorted(feedback_ratings)
index = int(len(sorted_ratings) * percentile)

# Print the 75th percentile
print(sorted_ratings[index])


# 11. Scenario: A quality control department wants to test the weight consistency of a product. The weights of a sample of products are as follows:
#     [10.2, 9.8, 10.0, 10.5, 10.3, 10.1]
#     Perform a hypothesis test to determine if the mean weight differs significantly from 10 grams.
# 

# In[6]:


import numpy as np
import scipy.stats as stats

# Create the data
weights = [10.2, 9.8, 10.0, 10.5, 10.3, 10.1]

# Calculate the sample mean and standard deviation
mean = np.mean(weights)
std = np.std(weights)

# Calculate the t-statistic
t = (mean - 10) / std

# Calculate the p-value
p_value = stats.t.sf(t, len(weights) - 1)

# Print the results
print('t-statistic:', t)
print('p-value:', p_value)


# 12. Scenario: A company wants to analyze the click-through rates of two different website designs. The number of clicks for each design is as follows:
#     Design A: [100, 120, 110, 90, 95]
#     Design B: [80, 85, 90, 95, 100]
#     Perform a chi-square test to determine if there is a significant difference in the click-through rates between the two designs.
# 

# In[8]:


expected_value = (total_clicks * design_proportion) / 2


# In[ ]:





# In[7]:


import numpy as np
import scipy.stats as stats

# Create the data
clicks_a = [100, 120, 110, 90, 95]
clicks_b = [80, 85, 90, 95, 100]

# Calculate the expected values
total_clicks = len(clicks_a) + len(clicks_b)
design_proportions = [0.6, 0.4]
expected_values = (total_clicks * design_proportions) / 2

# Calculate the chi-square statistic
chi_square = 0
for clicks, expected in zip(clicks_a, expected_values):
  chi_square += ((clicks - expected)**2) / expected

# Calculate the p-value
p_value = stats.chi2.sf(chi_square, len(clicks_a) - 1)

# Print the results
print('chi-square statistic:', chi_square)
print('p-value:', p_value)


# 3. Scenario: A survey is conducted to measure customer satisfaction with a product on a scale of 1 to 10. The data collected is as follows:
#     [7, 9, 6, 8, 10, 7, 8, 9, 7, 8]
#     Calculate the 95% confidence interval for the population mean satisfaction score.
# 

# In[9]:


import statistics
import numpy as np

# Create the data
satisfaction_scores = [7, 9, 6, 8, 10, 7, 8, 9, 7, 8]

# Calculate the sample mean and standard deviation
mean = statistics.mean(satisfaction_scores)
std = statistics.stdev(satisfaction_scores)

# Calculate the confidence interval
confidence_interval = 1.96 * std / np.sqrt(len(satisfaction_scores))

# Print the confidence interval
print('Confidence interval:', mean - confidence_interval, mean + confidence_interval)


# 14. Scenario: A company wants to analyze the effect of temperature on product performance. The data collected is as follows:
#     Temperature (in degrees Celsius): [20, 22, 23, 19, 21]
#     Performance (on a scale of 1 to 10): [8, 7, 9, 6, 8]
#     Perform a simple linear regression to predict performance based on temperature.
# 

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create the data
temperature = np.array([20, 22, 23, 19, 21])
performance = np.array([8, 7, 9, 6, 8])

# Create the linear regression model
model = LinearRegression()
model.fit(temperature, performance)

# Print the coefficients
print(model.coef_)
print(model.intercept_)

# Plot the data and the regression line
plt.scatter(temperature, performance)
plt.plot(temperature, model.predict(temperature))
plt.show()


# 15. Scenario: A study is conducted to compare the preferences of two groups of participants. The preferences are measured on a Likert scale from 1 to 5. The data collected is as follows:
#     Group A: [4, 3, 5, 2, 4]
#     Group B: [3, 2, 4, 3, 3]
#     Perform a Mann-Whitney U test to determine if there is a significant difference in the median preferences between the two groups.
# 

# In[11]:


import numpy as np
import scipy.stats as stats

# Create the data
group_a = [4, 3, 5, 2, 4]
group_b = [3, 2, 4, 3, 3]

# Perform the Mann-Whitney U test
u_statistic, p_value = stats.mannwhitneyu(group_a, group_b)

# Print the results
print('U-statistic:', u_statistic)
print('p-value:', p_value)


# 16. Scenario: A company wants to analyze the distribution of customer ages. The data collected is as follows:
#     [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
#     Calculate the interquartile range (IQR) of the ages.
# 

# In[12]:


import statistics

# Create the data
ages = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# Calculate the quartiles
first_quartile = statistics.percentileofscore(ages, 25)
third_quartile = statistics.percentileofscore(ages, 75)

# Calculate the IQR
iqr = third_quartile - first_quartile

# Print the IQR
print(iqr)


# 17. Scenario: A study is conducted to compare the performance of three different machine learning algorithms. The accuracy scores for each algorithm are as follows:
#     Algorithm A: [0.85, 0.80, 0.82, 0.87, 0.83]
#     Algorithm B: [0.78, 0.82, 0.84, 0.80, 0.79]
#     Algorithm C: [0.90, 0.88, 0.89, 0.86, 0.87]
#     Perform a Kruskal-Wallis test to determine if there is a significant difference in the median accuracy scores between the algorithms.
# 

# In[13]:


import numpy as np
import scipy.stats as stats

# Create the data
algorithm_a = [0.85, 0.80, 0.82, 0.87, 0.83]
algorithm_b = [0.78, 0.82, 0.84, 0.80, 0.79]
algorithm_c = [0.90, 0.88, 0.89, 0.86, 0.87]

# Perform the Kruskal-Wallis test
h_statistic, p_value = stats.kruskal(algorithm_a, algorithm_b, algorithm_c)

# Print the results
print('H-statistic:', h_statistic)
print('p-value:', p_value)


# 18. Scenario: A company wants to analyze the effect of price on sales. The data collected is as follows:
#     Price (in dollars): [10, 15, 12, 8, 14]
#     Sales: [100, 80, 90, 110, 95]
#     Perform a simple linear regression to predict
# 
#  sales based on price.
# 

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create the data
price = np.array([10, 15, 12, 8, 14])
sales = np.array([100, 80, 90, 110, 95])

# Create the linear regression model
model = LinearRegression()
model.fit(price, sales)

# Print the coefficients
print(model.coef_)
print(model.intercept_)

# Plot the data and the regression line
plt.scatter(price, sales)
plt.plot(price, model.predict(price))
plt.show()


# 9. Scenario: A survey is conducted to measure the satisfaction levels of customers with a new product. The data collected is as follows:
#     [7, 8, 9, 6, 8, 7, 9, 7, 8, 7]
#     Calculate the standard error of the mean satisfaction score.
# 
# 

# In[15]:


import statistics

# Create the data
satisfaction_scores = [7, 8, 9, 6, 8, 7, 9, 7, 8, 7]

# Calculate the mean
mean = statistics.mean(satisfaction_scores)

# Calculate the standard deviation
standard_deviation = statistics.stdev(satisfaction_scores)

# Calculate the standard error of the mean
standard_error_of_mean = standard_deviation / np.sqrt(len(satisfaction_scores))

# Print the standard error of the mean
print(standard_error_of_mean)


# 20. Scenario: A company wants to analyze the relationship between advertising expenditure and sales. The data collected is as follows:
#     Advertising Expenditure (in thousands): [10, 15, 12, 8, 14]
#     Sales (in thousands): [25, 30, 28, 20, 26]
#     Perform a multiple regression analysis to predict sales based on advertising expenditure.
# 
# 

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create the data
advertising_expenditure = np.array([10, 15, 12, 8, 14])
sales = np.array([25, 30, 28, 20, 26])

# Create the multiple regression model
model = LinearRegression()
model.fit(advertising_expenditure, sales)

# Print the coefficients
print(model.coef_)
print(model.intercept_)

# Plot the data and the regression line
plt.scatter(advertising_expenditure, sales)
plt.plot(advertising_expenditure, model.predict(advertising_expenditure))
plt.show()


# In[ ]:




