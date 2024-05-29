#!/usr/bin/env python
# coding: utf-8

# ## Optimizing Vehicle Fuel Efficiency: Predictive Modeling of MPG(Miles Per Gallon)

# ##### By: Yordanos Simegnew Muche

# ##### Abstract

# This project delves into a comprehensive linear regression analysis of a dataset focusing on vehicle miles per gallon (MPG). The methodology encompasses a structured approach, beginning with data preprocessing steps such as standardizing column names, handling missing values, and addressing outliers. Following data preparation, an exploratory data analysis (EDA) is conducted, comprising univariate, bivariate, and multivariate analyses to gain insights into the relationships between variables. 
# 
# Subsequently, a linear regression model is trained on the dataset to predict MPG values, followed by model evaluation and visualization. The evaluation highlights the model's goodness of fit, as indicated by an (R^2) value of 0.821, signifying a strong correlation between predicted and actual MPG values. 
# 
# The scatter plot depicting actual versus predicted values illustrates a tight clustering around the diagonal line, indicating accurate predictions overall. However, analysis of residuals reveals outliers that warrant further investigation to refine model performance. Despite the model's effectiveness, opportunities for enhancement are identified, including fine-tuning, feature engineering, or exploring more complex models. 
# 
# Overall, this study provides valuable insights into the predictive capabilities of linear regression in the context of vehicle MPG, offering implications for improving fuel efficiency and informing decision-making in the automotive industry.

# ##### Project outline

# 1. Importing the necessary libraries
# 2. Loading the dataset
# 3. First Look at the MPG dataset
# 4. Data Cleaning
#     >1. Standardizing column names
#     >2. Standardizing text columns
#     >3. Removing unecessary spaces from text columns (if any)
#     >4. Removing duplicated records (if any)
#     >5. Handling Missing values
#     >6. Extracting information from from a column
#     >7 Handling an outlier records
# 5. Explotatory data analysis(EDA)
#     >1. Univariate Analysis
#     >2. Bivariate Analysis
#     >3. Multivariate Analysis
# 6. Model trianing
# 7. Model Prediction
# 8. Model Evaluation
# 9. Model Visualization

# ### 1. Importing Necessary Libraries

# In[1]:


# Importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)


# ### 2. Loading Our Dataset

# In[2]:


# loading the mpg dataset from seaborn library
df = sns.load_dataset("mpg")


# In[3]:


# displaying the first 5 records in mpg dataset
df.head()


# ### 3. First Look at Our Data

# In[4]:


# displaying 3 sample records from mpg dataset.
df.sample(3)


# In[5]:


# checking for the dimensions of mpg dataset
df.shape


# In[6]:


# displaying the general inforamation of the mpg dataset.
df.info()


# ### 4. Data Cleaning

# ##### 1. Standardizing column names

# 1. Renaming the Columns such that they appear as a title case format
# 2. Adding units to each of the columns that need units

# In[7]:


# displaying the column indices
df.columns


# In[8]:


# defining a funcion that can make the column names title case
def title_maker(column_names):
    return column_names.str.title()


# In[9]:


# applying the title_maker function to the columns
title_maker(df.columns)


# In[10]:


# overwritting the column names
df.columns = title_maker(df.columns)


# In[11]:


# displaying 2 sample records
df.sample(2)


# Adding units to each features
# 1. displacement is given in cubic inches(in^3)
# 2. horsepower(HP)
# 3. weight is given in pounds(lbs)
# 4. acceleration is given in meter per second squared(m/s^2)
# 

# In[12]:


# renaming the column names
df.rename(columns ={"Displacement":"Displacement(in^3)","Horsepower":"Horsepower(HP)","Weight":"Weight(lbs)","Acceleration":"Acceleration(m/s^2)"},inplace = True)


# In[13]:


# displaying the column names
df.columns


# ##### 2. Standardizing Text columns

# Lets standardize our text columns (by making them all title case)
# 1. Origin
# 2. Name

# In[14]:


# displaying unique origin names
df.Origin.unique()


# In[15]:


# checking for the general information of the dataframe
df.info()


# In[16]:


# defining a function that can make the text columns to title case
def text_maker(column):
    return column.str.title()


# In[17]:


# applying the text maker function and overwritting the text columns
df[df.select_dtypes("object").columns] = df.select_dtypes("object").apply(text_maker)


# In[18]:


# displaying 3 sample records.
df.sample(3)


# ##### 3. Removing Unecessary Spaces from text columns (if any)

# In[19]:


# defining a space remover function that can remove space from a text columns.
def space_remover(columns):
    return columns.str.strip()


# In[20]:


# applying the text remover function and overwrritting the text columns.
df[df.select_dtypes("object").columns] = df.select_dtypes("object").apply(space_remover)


# In[21]:


# displaying sample 2 records
df.sample(2)


# ##### 4. Removing Duplicated Columns (if any)

# In[22]:


# first let's check for duplicated columns
df.duplicated().any()


# In[23]:


df.duplicated().sum()


# In[24]:


# from the above two codes we can see that, we don't have any duplicated columns.


# ##### 5. Handling Missing Values(if any)

# In[25]:


# for the case of this project, we handle missing values by removing the record with missing value from our dataset.
df.isnull().any()


# In[26]:


# we have missing values in our horsepower column, let's see how many missing values we have.
df.isnull().sum()


# In[27]:


# let's see the records with missing values
df[df.isnull().any(axis = 1)]


# In[28]:


# now let's remove this missing values from our dataset
df[df.isnull().any(axis = 1)].index


# In[29]:


# one of the two methods below work
# df.drop(df[df.isnull().any(axis = 1)].index)
df.dropna(inplace = True)


# In[30]:


# let's check the general info of the dataframe.
df.info()


# In[31]:


# again let's check for the shape of the dataframe
df.shape


# In[32]:


# now let's reset the index
df.reset_index(inplace = True)


# In[33]:


# display sample record
df.sample()


# In[34]:


# now let's drop the index column
df.drop("index", axis = 1, inplace = True)


# In[35]:


# displaying 2 sample records
df.sample(2)


# In[36]:


# let's check for the general information of the dataframe
df.info()


# ##### 6. extracting information from a column

# In[37]:


# HERE first let's fix the model_year column and let's extract the age information from the model_year column
df.Model_Year = df.Model_Year + 1900


# In[38]:


# the model_year column
df.Model_Year


# In[39]:


# displaying the today's date
today = datetime.today()


# In[40]:


# year of today
today.year


# In[41]:


# inserting the age column next to the model_year column
df.insert(df.columns.get_loc("Model_Year")+1,"Age", (today.year - df.Model_Year))


# In[42]:


# displaying the first 5 records
df.head()


# #### 7. Handling an Outliers

# In[43]:


# first let's see the outlier limits 
# for our use case we assign an outlier records as records which are over or below 3 stadard deviations from the mean.
def outlier_limits(df):
    upper_limit_list = []
    lower_limit_list = []
    for column in df.select_dtypes(["int64","float64"]).columns:
        mean = df[column].mean()
        std = df[column].std()
        upper_limit = mean + 3*std
        lower_limit = mean - 3*std
        upper_limit_list.append(upper_limit)
        lower_limit_list.append(lower_limit)
        
    return pd.DataFrame((upper_limit_list, lower_limit_list), columns = df.select_dtypes(["int64","float64"]).columns)


# In[44]:


# outlier limits
outlier_limits(df)


# In[45]:


# now let's see the outlier records
def return_outlier(column):
    outliers_df = pd.DataFrame()
    for column in df.select_dtypes(["int64","float64"]).columns:
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        filtered_df = df[(df[column] > upper_bound) |(df[column] < lower_bound)]
        outliers_df = pd.concat([outliers_df,filtered_df])
    return outliers_df


# In[46]:


# outlier records
return_outlier(df)


# In[47]:


# shape of the outliers dataframe
return_outlier(df).shape


# In[48]:


# we have 7 records with an outliers so we handle this outliers by removing this form our dataset.
return_outlier(df).index


# In[49]:


# the index of the outliers dataframe
index = return_outlier(df).index


# In[50]:


# removing the outlier records
df.drop(index, inplace = True)


# In[51]:


# checking the general information of the dataframe
df.info()


# In[52]:


# reset the index of the dataframe
df.reset_index(inplace = True)


# In[53]:


# displaying sample 3 records
df.sample(3)


# In[54]:


# removing the index column
df.drop("index",inplace = True, axis = 1)


# In[55]:


# displaying 1 sample record
df.sample(1)


# In[56]:


# checking for the general information of the dataframe
df.info()


# ### 5. Exploratory Data Analysis

# ##### 5.1 Univariate Analysis

# In[57]:


# 1. Mpg analysis
plt.figure(figsize = (8,4))
sns.distplot(df.Mpg)
plt.title("mpg_distribution")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\1.mpg_distribution.jpg", format = "jpg")
plt.show()


# As we can see from the mpg distribution plot it is 'right-skewed' and we can conclude the following points from the above plot:
# 1. The majority of vehicles tends to have lower Mpg value while smaller number of vehiclses have higher mpg values.
# 2. Majority of the vehicles have an mpg value between 15 and 20.
# 3. Even if most vehicles have moderate fuel efficiency, there are few vehicles with exceptionally higher mpg.
# 4. The mean mpg is higher than the median of mpg
# 

# In[58]:


df.Mpg.mean()


# In[59]:


df.Mpg.median()


# In[60]:


# 2. cylinders analysis
plt.figure(figsize = (8,4))
sns.countplot(data = df,x = df.Cylinders)
plt.title("Number_of_Cylinedrs")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\2.Number_of_Cylinedrs.jpg", format = "jpg")
plt.show()


# We can draw the following points from the above bar plot for number of cylinders
# 1. The most common type of engine amont the vehicles in the dataset is 4 cylinders.
# 2. The 6 and 8 cylinder engines, which typically found in larger and more powerfull vehicles are the next most appearing type of cylinders in the dataset.
# 3. Very few vehicles consist the 3 and 5 cylinder configurations.

# In[61]:


# 3. Horse Power analysis
plt.figure(figsize = (8,4))
sns.distplot(df['Horsepower(HP)'])
plt.title("Distribution_of_Horsepower")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\3.Distribution_of_Horsepower.jpg", format = "jpg")
plt.show()


# In[62]:


df["Horsepower(HP)"].mean()


# In[63]:


df["Horsepower(HP)"].median()


# From the above density plot of horsepower we can conclude that:
# 1. The plot is "right skewed" means most of the vehicles have the horsepower less than that of the mean horsepower.
# 2. Most vehicles have horsepower between 85 and 95
# 3. The mean horsepower is greater than that of the median horsepower

# In[64]:


# 4. Origins Analysis
plt.figure(figsize = (8,3))
sns.countplot(data = df, x = df.Origin)
plt.title("Count of vehicles by origin")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\4.Count of vehicles by origin.jpg", format = "jpg")
plt.show()


# From the above bar plot of origins we can conclude that:
# 1. Most of the vehicles are from usa.
# 2. Vehicles from japan and europe have almost equal numbers.

# ##### 5.2 Bivariate Analysis

# In[65]:


# 1. Mpg vs Displacement
plt.figure(figsize = (8,4))
sns.scatterplot(data = df, y = df["Displacement(in^3)"], x= df.Mpg)
plt.title("Mpg vs Displacement")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\5.Mpg vs Displacement.jpg", format = "jpg")
plt.show()


# AS the mpg variable increases the engine displacement decreases. MEANS mpg and displacement have negative coorelation or inverse correlation.

# In[66]:


# 2. Mpg vs Horsepower
plt.figure(figsize = (8,4))
sns.lineplot(data = df, y = df["Horsepower(HP)"], x= df.Mpg)
plt.title("Mpg vs Horsepower")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\6. Mpg vs Horsepower.jpg", format = "jpg")
plt.show()


# Vehicles with highest Horsepower have lower mpg and viceversa. MEANS mpg and horsepower also have negative correlation or inverse correlation.

# In[67]:


# 3. Mpg vs Acceleration
plt.figure(figsize = (8,4))
sns.jointplot(data = df, y = df['Acceleration(m/s^2)'], x= df.Mpg , kind = "reg")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\7. Mpg vs Acceleration.jpg", format = "jpg")
plt.show()


# As the vehicles mpg increases the Acceleration also increases. mpg and acceleration have positive correlation.

# In[68]:


# 4. Mpg vs Weight
plt.figure(figsize = (8,4))
sns.jointplot(data = df, y = df["Weight(lbs)"], x= df.Mpg , kind = "kde")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\8. Mpg vs Weight.jpg", format = "jpg")
plt.show()


# The majourity of the data lies in that region and the mpg have an inverse coorelation with the weight 

# In[69]:


# 5. Mpg vs Origin
plt.figure(figsize = (8,4))
sns.boxplot(data = df, y =df.Mpg, x= df.Origin )
plt.title("Mpg vs Origin")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\9. Mpg vs Origin.jpg", format = "jpg")
plt.show()


# From the above plot we can conclude that:
# 1. The median of mpg for vehicles from japan is greater than that of median of mpg(usa and europe)
# 2. The least mpg belongs to usa and the maximum mpg belongs to japan
# 3. More mpg outliers exist in europe.(evenif we set outliers as 3 standardeviations from the mean.)
# 4. The mpg of vehicles from europe falls around the mean compared to that of usa and japan.

# ##### 5.3 Multivariate Analysis

# In[70]:


# 5. Mpg vs Horsepower vs Origin
plt.figure(figsize = (8,4))
sns.lmplot(data = df, x= "Mpg", y ="Horsepower(HP)", hue = "Origin")
plt.title("Mpg vs Horsepower vs Origin")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\10. Mpg vs Horsepower vs Origin.jpg", format = "jpg")
plt.show()


# From the above linear model plot we can conclude that 
# 1. For all vehicles from the three regions as the mpg increaases the horsepower decreases.
# 2. The decrease in horsepower as the mpg increase is, more in vehicles from usa.
# 3. Vehicles from usa have highest amount of horsepower

# In[71]:


# 5. Mpg vs Displacement vs Origin
plt.figure(figsize = (8,4))
sns.lmplot(data = df, x= "Mpg", y ="Displacement(in^3)", hue = "Origin")
plt.title(" Mpg vs Displacement vs Origin")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\11. Mpg vs Displacement vs Origin.jpg", format = "jpg")
plt.show()


# From the above linear model plot we can clearly see that
# 1. Vehicles with highest amount of engine displacement belongs to usa
# 2. For all vehicles from europe, japan and usa as mpg increases the engine displacement decreases.
# 3. The decrease in displacement for vehicles from usa is higher than the decrease in displacement for vehicles from japan and europe as mpg increases.

# In[72]:


# correlation plot
plt.figure(figsize = (10,4))
sns.heatmap(df.select_dtypes(["int64","float64"]).corr(),annot = True, linewidth = 0.5,  cmap = "Spectral")
plt.title("correlation plot")
plt.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\12. correlation plot.jpg", format = "jpg")
plt.show()


# As we can see from the above heatmap plot 
# 1. The acceleration have the least coorelation with the mpg so we have to remove it to prevent our model from overfitting.
# 2. The model year and age variables have same result so we have to remove one of them.
# 3. The displacement and cylinders variable are highly coorrelated, on the same way they are highly coorelated with the mpg variable so here we have to remove the displacement variable to prevent overfitting.

# The name variable does not have any impact in our prediction analysis using linear regression model so we also remove the name variable form our dataframe

# In[73]:


df.drop(["Model_Year","Acceleration(m/s^2)","Displacement(in^3)","Name"], axis = 1, inplace = True)


# In[74]:


df.sample(4)


# In[75]:


# here we have to assign the origin variables as a numbers 
df  = pd.get_dummies(df,dtype = "int",drop_first = True)


# In[76]:


df.sample(3)


# ### 6. Model Training

# In[77]:


# now lets separate our target label and features
x = df.drop("Mpg",axis = 1) # features
y = df.Mpg    # target labels


# In[78]:


# sample 3 records from the features
x.sample(3)


# In[79]:


# sample 3 records form the label
y.sample(3)


# Now let's split our data in to training and testing sets using train test split module from scikit learn using 15% of our data for trianing and 85% for testing.

# In[80]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 42, test_size = 0.15)


# In[81]:


# checking for shapes
x_train.shape


# In[82]:


# checking for shapes
x_test.shape


# In[83]:


# checking for shapes
y_train.shape


# In[84]:


# checking for shapes
y_test.shape


# In[85]:


# import a linear regression model
from sklearn.linear_model import LinearRegression


# In[86]:


# assigning the linear regression model
mpg_model = LinearRegression()


# In[87]:


# mpg_model
mpg_model


# In[88]:


# Train the model using the training data
mpg_model.fit(x_train,y_train)


# In[89]:


# Get the intercept (bias term) of the trained model (y_intercept)
mpg_model.intercept_


# In[90]:


# Get the coefficients of the features from the trained model
mpg_model.coef_


# In[91]:


# assigning the index for the coefficients
index_c = df.drop("Mpg",axis = 1).columns
index_c


# In[92]:


# data frame of coefficients
pd.DataFrame(mpg_model.coef_, index = index_c, columns = ["Coefficient_Values"])


# ### 7. Model Prediction

# In[93]:


# Make predictions on the test data using the trained model and displaying predictions
prediction = mpg_model.predict(x_test)
prediction


# In[94]:


# data frame of actual and predicted values
my_dict = {"actual": y_test, "prediction" : prediction}
pd.DataFrame(my_dict)


# ### 8. Model Evaluation

# In[95]:


# Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[96]:


# defining a function for evaluation metrices
def metrics(actual,prediction):
    MAE = mean_absolute_error(actual,prediction)
    MSE = mean_squared_error(actual,prediction)
    R2_score = r2_score(actual,prediction)
    return MAE, MSE, R2_score


# In[97]:


# assingning index and column names
index_name = ["MAE","MSE","R2_score"]
column_name = ["values"]


# In[98]:


# data frame of evaluation metrices
pd.DataFrame(metrics(y_test,prediction), index = index_name, columns = column_name)


# ### 9. Model Visualization

# In[99]:


# Linear Regression model visualization using yellowbrick regressor
from yellowbrick.regressor import PredictionError
visualizer = PredictionError(mpg_model)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.show(outpath = "C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\13. linear_regression_prediction_error.jpg", format='jpg')


# Here are some key points and conclusions from the prediction error plot above
# 
# 1. Goodness of Fit (( R^2 ) Value): The ( R^2) value is 0.821, indicating that approximately 82.1% of the variance in the target variable( y ) can be explained by the model. This suggests a strong correlation between the predicted values and the actual values, implying the model is performing well.
# 
# 2. Prediction Accuracy: The scatter plot of the actual values versus the predicted values shows that most of the points are closely clustered around the diagonal line, which represents perfect predictions. This further indicates that the model's predictions are generally accurate.
# 
# 3. Residuals and Error Analysis: The points deviating from the diagonal line represent prediction errors. While most points are near the line, some outliers are further away, indicating areas where the model's predictions are less accurate. These outliers could be investigated further to understand why the model is less accurate for these instances (e.g., potential anomalies or areas where the model might be underfitting or overfitting).
# 
# 4. Line of Best Fit: The dashed line represents the line of best fit for the predicted versus actual values. Since this line is close to the identity line (the 45-degree diagonal), it further confirms that the model's predictions are in good agreement with the actual values.
# 
# 5. Model Evaluation: Overall, the plot and the (R^2) value suggest that the linear regression model performs well for the given data. However, the presence of some outliers indicates there is room for improvement, possibly through further tuning of the model, feature engineering, or trying more complex models.

# In[100]:


from yellowbrick.regressor import ResidualsPlot

visualizer = ResidualsPlot(mpg_model)
visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.show(outpath = "C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\14.linear_regression_Residual_plots.jpg", format = "jpg")


# In[101]:


# Linear Regression model visualization using yellowbrick regressor
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(mpg_model)
visualizer.fit(x_train,y_train)
visualizer.score(x_test,y_test)
visualizer.show()
visualizer.savefig("C:\\Users\\yozil\\Desktop\\mpg_project\\plots\\14.linear_regression_Residual_plots.jpg", format='jpg')
plt.show()


# Here are the main conclusions from the above residuals plot
# 
# 1. Model Fit (R² values):
#    - Both the training and test sets have an R² value of 0.821, indicating that the model explains 82.1% of the variance in the target variable for both sets. This suggests that the model has a good fit and is consistent across both datasets, indicating no overfitting.
# 
# 2. Residual Distribution:
#    - The residuals (the difference between the actual values and the predicted values) are plotted against the predicted values. For a good linear regression model, the residuals should be randomly scattered around the horizontal axis (residuals = 0) without any distinct pattern.
#    - The residuals in this plot do not appear to show a strong pattern, suggesting that the model is capturing the underlying relationship between the features and the target variable reasonably well. However, there is some noticeable heteroscedasticity (variance of residuals increasing with predicted values), which might need further investigation.
# 
# 3. Normality of Residuals:
#    - The distribution of the residuals on the right-hand side histogram should ideally be normally distributed (bell-shaped). The histogram indicates that while the residuals are roughly centered around zero, their distribution is slightly skewed, which might suggest some non-normality in the residuals.
# 
# 4. Outliers and Leverage Points:
#    - There are a few points with high residuals (far from the horizontal axis) which might be outliers or high-leverage points. These points can have a disproportionate impact on the model's predictions and may need to be investigated further.
# 
# 5. Bias:
#    - The plot does not show a systematic bias (residuals are not consistently above or below the zero line) across the range of predicted values. This implies that the model does not systematically overestimate or underestimate the target variable.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




