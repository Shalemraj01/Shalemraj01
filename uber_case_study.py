#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://upload.wikimedia.org/wikipedia/commons/d/d6/UBER_%281%29.jpg" width="800" height="400"></center>
# 
# <b><h2><center>Uber Case Study</center></h2></b>

# ### **Context:**
# Ridesharing is a service that arranges transportation on short notice. It is a very volatile market and its demand fluctuates wildly with time, place, weather, local events, etc. The key to being successful in this business is to be able to detect patterns in these fluctuations and cater to the demand at any given time.  
# 
# ### **Objective:**
# Uber Technologies, Inc. is an American multinational transportation network company based in San Francisco and has operations in over 785 metropolitan areas with over 110 million users worldwide. As a newly hired Data Scientist in Uber's New York Office, you have been given the task of extracting actionable insights from data that will help in the growth of the the business. 
# 
# ### **Key Questions:**
# 1. What are the different variables that influence the number of pickups?
# 2. Which factor affects the number of pickups the most? What could be the possible reasons for that?
# 3. What are your recommendations to Uber management to capitalize on fluctuating demand?
# 
# ### **Data Description:**
# The data contains the details for the Uber rides across various boroughs (subdivisions) of New York City at an hourly level and attributes associated with weather conditions at that time.
# 
# * pickup_dt: Date and time of the pick-up
# * borough: NYC's borough
# * pickups: Number of pickups for the period (hourly)
# * spd: Wind speed in miles/hour
# * vsb: Visibility in miles to the nearest tenth
# * temp: Temperature in Fahrenheit
# * dewp: Dew point in Fahrenheit
# * slp: Sea level pressure
# * pcp01: 1-hour liquid precipitation
# * pcp06: 6-hour liquid precipitation
# * pcp24: 24-hour liquid precipitation
# * sd: Snow depth in inches
# * hday: Being a holiday (Y) or not (N)

# ### **Importing the necessary libraries**

# In[2]:


# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### **Loading the dataset**

# In[3]:


# #Mount drive
# from google.colab import drive
# drive.mount('/content/drive')


# In[4]:


data = pd.read_csv(r"C:\Users\shale\Downloads\Uber_Data new.csv")


# In[5]:


df = data


# In[6]:


data.info()


# In[7]:


df2['pickup_dt'] = pd.to_datetime(df2['pickup_dt'])


# In[ ]:


df2.info()


# In[ ]:


df2.info()


# In[ ]:


data.info()


# In[ ]:


# copying data to another variable to avoid any changes to original data
df = data.copy()


# ### **Data Overview**

# The initial steps to get an overview of any dataset is to: 
# - observe the first few rows of the dataset, to check whether the dataset has been loaded properly or not
# - get information about the number of rows and columns in the dataset
# - find out the data types of the columns to ensure that data is stored in the preferred format and the value of each property is as expected.
# - check the statistical summary of the dataset to get an overview of the numerical columns of the data

# #### **Displaying the first few rows of the dataset**

# In[ ]:


# looking at head (5 observations) 
df.head()


# - *The `pickup_dt` column contains the date and time of pickup* 
# - *The `borough` column contains the name of the New York borough in which the pickup was made*
# - *The `pickups` column contains the number of pickups in the borough at the given time* 
# - *Starting from `spd` to `sd`, all the columns are related to weather and  are numerical in nature*
# - *The `hday` column indicates whether the day of the pickup is a holiday or not (Y: Holiday, N: Not a holiday)*

# #### **Checking the shape of the dataset**

# In[8]:


df.shape


# - *The dataset has 29,101 rows and 13 columns*

# #### Checking the data types of the columns for the dataset

# In[9]:


df.info()


# - *All the columns have 29,101 observations except `borough` and `temp` which has 26058 and 28742 observations indicating that there are some missing values in them*
# - *The `pickup_dt` column is being read as a 'object' data type but it should be in date-time format*
# - *The `borough` and `hday` columns are of object type while the rest of the columns are numerical in nature*
# - *The object type columns contain categories in them*

# #### **Getting the statistical summary for the dataset**

# In[10]:


df.describe()


# - *There is a huge difference between the 3rd quartile and the maximum value for the number of pickups (`pickups`) and snow depth (`sd`) indicating that there might be outliers to the right in these variables*
# - *The temperature has a wide range indicating that data consists of entries for different seasons*

# **Let's check the count of each unique category in each of the categorical/object type variables.**

# In[11]:


df['borough'].unique()


# - *We can observe that there are 5 unique boroughs present in the dataset for New York plus EWR (Newark Liberty Airport)*
# 
#     - *Valid NYC Boroughs (Bronx, Brooklyn, Manhattan, Queens, and Staten Island)*
#     - *EWR is the acronym for Newark Liberty Airport (EWR IS NOT AN NYC BOROUGH)* 
#         - *NYC customers have the flexibility to catch flights to either: (1) JFK Airport, LaGuardia Airport, or Newark Liberty Airport (EWR)*. 

# In[12]:


df['hday'].value_counts(normalize=True)*100


# In[13]:


df['hday'].value_counts()


# - *The number of non-holiday observations is much more than holiday observations which make sense*
# -*Around 96% of the observations are from non-holidays*
# 

# We have observed earlier that the data type for `pickup_dt` is object in nature. Let us change the data type of `pickup_dt` to date-time format.

# #### **Fixing the datatypes**

# In[14]:


pd.to_datetime(data['pickup_dt']).dt.day_name()


# In[15]:


# day starts from monday....in python


# In[16]:


pd.to_datetime(data['pickup_dt']).dt.dayofweek


# In[17]:


data['pickup_dt']


# In[18]:


data.info()


# In[19]:


pd.to_datetime(df['pickup_dt'])


# In[20]:


df['pickup_dt'] = pd.to_datetime(df['pickup_dt'], format="%d-%m-%Y %H:%M")


# In[21]:


df['dayinmonth'] = df['pickup_dt'].dt.daysinmonth


# In[22]:


df[['dayinmonth','pickup_dt']].drop_duplicates().values[:200]


# In[23]:


dir(df['pickup_dt'].dt)


# In[24]:


df['pickup_dt']


# Let's check the data types of the columns again to ensure that the change has been executed properly.

# In[25]:


df.info()


# - *The data type of the `pickup_dt` column has been succesfully changed to date-time format*
# - *There are now 10 numerical columns, 2 object type columns and 1 date-time column*

# Now let's check the range of time period for which the data has been collected.

# In[26]:


df['pickup_dt'].min() # this will display the date from which data observations have been started


# In[27]:


df['pickup_dt'].max() # this will display the last date of the dataset


# - *So the time period for the data is from Janunary to June for the year 2015*
# - *There is a significant difference in the weather conditions in this period which we have observed from our statistical summary for various weather parameters such as temperature ranging from 2F to 89F*

# Since the `pickup_dt` column contains the combined information in the form of date, month, year and time of the day, let's extract each piece of information as a separate entity to get the trend of rides varying over time.
# 

# #### **Extracting date parts from pickup date**

# In[28]:


# extract the year from the pickut date
#extracting the month name from the date
# extracting the hour from the time
# extracting the day from the date
#extracting the day of the week from the date


# In[29]:


# Replacing with mean: In this method the missing values are imputed with the mean of the column. Mean gets impacted by the presence of outliers, and in such cases where the column has outliers using this method may lead to erroneous imputations.

# Replacing with median: In this method the missing values are imputed with the median of the column.
#In cases where the column has outliers, 
#median is an appropriate measure of central tendency to deal with the missing values over mean.

# Replacing with mode: In this method the missing values are imputed with the mode of the column.
#This method is generally preferred with categorical data.


# In[ ]:





# In[ ]:





# In[30]:


# Extracting date parts from pickup date
df['start_year'] = df.pickup_dt.dt.year # extracting the year from the date
df['start_month'] = df.pickup_dt.dt.month_name() # extracting the month name from the date
df['start_hour'] = df.pickup_dt.dt.hour # extracting the hour from the time
df['start_day'] = df.pickup_dt.dt.day # extracting the day from the date
df['week_day'] = df.pickup_dt.dt.day_name() # extracting the day of the week from the date


# In[31]:


df.dtypes


# Now we can remove the `pickup_dt` column from our dataset as it will not be required for further analysis.

# In[32]:


# removing the pickup date column 
df.drop('pickup_dt',axis=1,inplace=True)


# **Let's check the first few rows of the dataset to see if changes have been applied properly**

# In[33]:


set1 = set(df['spd'])


# In[34]:


type(set1)


# In[35]:


dir(set1)


# In[36]:


set1.dtype


# In[37]:


set2 = set(df['vsb'])


# In[38]:


set1.intersection(set2)


# In[39]:


df.head()


# **We can see the changes have been applied to the dataset properly.**

# Let's analyze the statistical summary for the new columns added in the dataset.

# In[40]:


df.describe(include='all').T 
# setting include='all' will get the statistical summary for both the numerical and categorical variables.


# - *The collected data is from the year 2015* 
# - *It consists of data for 6 unique months*

# **We have earlier seen that the `borough` and `temp` columns have missing values in them. So let us see them in detail before moving on to do our EDA.**

# ### **Missing value treatment**

# One of the commonly used method to deal with the missing values is to impute them with the central tendencies - mean, median, and mode of a column.
# 
# * `Replacing with mean`: In this method the missing values are imputed with the mean of the column. Mean gets impacted by the presence of outliers, and in such cases where the column has outliers using this method may lead to erroneous imputations. 
# 
# * `Replacing with median`: In this method the missing values are imputed with the median of the column. In cases where the column has outliers, median is an appropriate measure of central tendency to deal with the missing values over mean.
# 
# * `Replacing with mode`: In this method the missing values are imputed with the mode of the column. This method is generally preferred with categorical data.

# Let's check how many missing values are present in each variable.

# In[41]:


# checking missing values across each columns
df.isnull().sum()


# - *The variable `borough` and `temp ` have 3043 and 359 missing values in them*
# - *There are no missing values in other variables*

# Let us first see the missing value of the `borough` column in detail.

# In[42]:


df['borough'].isnull().sum()


# In[43]:


len(df)


# In[44]:


len(df[df['borough'].isna()])


# In[45]:


df.notnull().sum()


# In[46]:


len(df['borough'].isna() == False)


# In[47]:


len(df['borough'] == "NaN")


# In[48]:


df['borough'].value_counts(dropna=False,normalize=True)


# In[49]:


df.describe(include='all')


# In[50]:


df['borough'].value_counts(normalize=True,dropna=False)*100


# In[51]:


# Checking the missing values further
df.borough.value_counts(normalize=True, dropna=False)


# - *All the 6 categories have the same percentage i.e. ~15%. There is no mode (or multiple modes) for this variable*
# - *The percentage of missing values is close to the percentage of observations from other boroughs*
# - *We can treat the missing values as a separate category for this variable*

# We can replace the null values present in the `borough` column with a new label as `Unknown`.

# In[52]:


type(df[df['borough'].isna()]['borough'].iloc[6])


# In[53]:


# Replacing NaN with Unknown
df['borough'].fillna('Unknown', inplace =True) 


# In[54]:


import numpy as np


# In[55]:


df[df['borough'] == np.nan]


# In[56]:


df['borough'].unique()


# *It can be observed that the new label `Unknown` has been added in the `borough` column*

# In[57]:


df.isnull().sum()


# The missing values in the `borough` column have been treated. Let us now move on to `temp` variable and see how to deal with the missing values present there.

# Since this is a numerical variable, so we can impute the missing values by mean or median but before imputation, let's analyze the `temp` variable in detail. 
# 
# Let us print the rows where the `temp` variable is having missing values.

# In[58]:


df.loc[df['temp'].isnull()==True]


# *There are 359 observations where `temp` variable has missing values. From the overview of the dataset, it seems as if the missing temperature values are from the Brooklyn borough in the month of January.* 
# 
# So let's confirm our hypothesis by printing the unique boroughs and month names present for these missing values.

# In[59]:


df[['temp','borough']]


# In[60]:


data[data['temp'].isna()]['borough'].unique()


# In[61]:


data.loc[data['temp'].isnull() == True,'borough'].value_counts()


# In[62]:


data.loc[data['temp'].isna(),'borough'].value_counts()


# In[63]:


df[df['temp'].isnull()==True]['borough']


# In[64]:


df.loc[df['temp'].isnull()==True,'borough'].value_counts()


# In[65]:


df.loc[df['temp'].isnull()==True,'start_month'].value_counts()


# *The missing values in `temp` are from the Brooklyn borough and they are from the month of January.* 
# 
# Let's check on which the date for the month of  January, missing values are present.

# In[66]:


df.loc[df['temp'].isnull()==True,'start_day'].unique() # days for which missing values are present


# In[67]:


df.columns


# In[68]:


df.loc[df['start_month']=='January', 'start_day'].unique() # unique days in the month of January


# *It can be observed  that out of the 31 days in January, the data is missing for the first 15 days.*

# Since from the statistical summary, the mean and median values of temperature are close to each other, hence we can impute the missing values in the `temp` column by taking the mean tempertaure of the Brooklyn borough during 16th to 31st January.

# We will use fillna() function to impute the missing values.
# 
# **fillna() -** The fillna() function is used to fill NaN values by using the provided input value.
# 
#        Syntax of fillna():  data['column'].fillna(value = x)
# 

# In[69]:


df['temp'] = df['temp'].fillna(value=df.loc[df['borough'] == 'Brooklyn','temp'].mean())


# In[70]:


df.isnull().sum()


# - *All the missing values have been imputed and there are no missing values in our dataset now*.

# Let's now perform the Exploratory Data Analysis on the dataset

# ### **Exploratory Data Analysis**

# ### **Univariate Analysis**

# **Let us first explore the numerical variables.**

# Univariate data visualization plots help us comprehend the descriptive summary of the particular data variable. These plots help in understanding the location/position of observations in the data variable, its distribution, and dispersion.
# 
# We can check the distribution of observations by plotting **Histograms** and **Boxplots**
# 
# A histogram takes as input a numeric variable only. The variable is cut into several bins, and the number of observations per bin is represented by the height of the bar
# 
# ![Screenshot 2022-02-26 145921.jpg](data:image/jpeg;base64,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)
# 
# 

# A boxplot gives a summary of one or several numeric variables. The line that divides the box into 2 parts represents the median of the data. The end of the box shows the upper and lower quartiles. The extreme lines show the highest and lowest value excluding outliers.
# 
# ![Screenshot 2022-02-26 150041.jpg](data:image/jpeg;base64,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)

# In[104]:


m = data['temp'].mean()


# In[105]:


data['temp'].fillna(data['temp'].mean(),inplace=True)


# In[106]:


data.isnull().sum()


# In[107]:


# Defining the function for creating boxplot and hisogram 
def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize)  # creating the 2 subplots
    
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="mediumturquoise")  # boxplot will be created and a star will indicate the mean value of the column
    
    if bins:
      sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, color="mediumpurple")
    else: 
      sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, color="mediumpurple")  # For histogram
    
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")  # Add mean to the histogram
    
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")  # Add median to the histogram


# #### **Observations on Pickups**

# In[108]:


histogram_boxplot(df,'pickups')


# - *The distribution of pickups is highly right skewed*
# - *There are a lot of outliers in this variable* 
# - *While mostly the number of pickups are at a lower end, we have observations where the number of pickups went as high as 8000*

# ####  **Observations on Visibility**

# In[109]:


histogram_boxplot(df,'vsb')


# - *The `visibility` column is is left-skewed*
# - *Both the mean and median are high, indicating that the visibility is good on most days*
# - *There are, however, outliers towards the left, indicating that visibility is extremely low on some days*
# - *It will be interesting to see how visibility affects the Uber pickup frequency*

# #### **Observations on Temperature**

# In[77]:


histogram_boxplot(df,'temp')


# - *Temperature does not have any outliers*
# - *50% of the temperature values are less than 45F (~7 degree celcius), indicating cold weather conditions*

# #### **Observations on Dew point**

# In[78]:


histogram_boxplot(df,'dewp')


# - *There are no outliers for dew point either*
# - *The distribution is similar to that of temperature. It suggests possible correlation between the two variables*
# - *Dew point is an indication of humidity, which is correlated with temperature*

# #### **Observations on Sea level pressure**

# In[79]:


histogram_boxplot(df,'slp')


# - *Sea level pressure distribution is close to normal*
# - *There are a few outliers on both the ends*

# ####  **Observations on Liquid Precipitation (Rain)**

# **1 hour liquid precipitation**

# In[80]:


histogram_boxplot(df,'pcp01')


# **6 hour liquid precipitation**

# In[81]:


histogram_boxplot(df,'pcp06')


# **24 hour liquid precipitation**

# In[82]:


histogram_boxplot(df,'pcp24')


# - *It rains on relatively fewer days in New York*
# - *Most of the days are dry*
# - *The outliers occur when it rains heavily*

# #### **Observations on Snow Depth**

# In[83]:


histogram_boxplot(df,'sd')


# - *We can observe that there is snowfall in the time period that we are analyzing*
# - *There are outliers in this data*
# - *We will have to see how snowfall affects pickups. We know that very few people are likely to get out if it is snowing heavily, so our pickups is most likely to decrease when it snows* 

# **Let's explore the categorical variables now**

# Bar Charts can be used to explore the distribution of Categorical Variables. Each entity of the categorical variable is represented as a bar. The size of the bar represents its numeric value.
# 
# 
# ![Screenshot 2022-02-26 150512.jpg](data:image/jpeg;base64,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)

# #### **Observations on holiday**

# In[84]:


sns.countplot(data=df,x='hday');


# - *The number of pickups is more on non-holidays than on holidays*

# #### **Observations on borough**

# In[85]:


plt.figure(figsize=(2,5))
sns.countplot(data=df,x='borough');

plt.xticks(rotation = 90);


# - *The observations are uniformly distributed across the boroughs except the observations that had NaN values and were attributed to Unknown borough*

# ### **Bivariate Analysis**

# Bi means two and variate means variable, so here there are two variables. The analysis is related to the relationship between the two variables.
# 
# Different types of Bivariate Analysis that can be done:
# - Bivariate Analysis of two Numerical Variables
# - Bivariate Analysis of two Categorical Variables
# - Bivariate Analysis of one Numerical Variables and one Categorical Variable

# **Let us plot bivariate charts between variables to understand their interaction with each other.**

# #### **Correlation by Heatmap**

# 
# A **heatmap** is a graphical representation of data as a color-encoded matrix. It is a great way of representing  the correlation for each pair of columns in the data.The *heatmap()* function of seaborn helps us to create such a plot
# 
# ![Screenshot 2022-02-26 153546.jpg](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCACvAMgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9U6KKKACiiigAooooAKKKKACimtIqfeYL9TVa6uJdmLZ7cN6yscfkKALdFZF5bX91pHli9jhui2TNHkLjPQVzc3g6/uDmTVo5D/tOxoA7uiuZ0nw1JZaLf2jXMUjXHR1zgcY5rJj8C3sRymoxIf8AZLCgDvKKwvD+l3+ntL9p1D7UhXCLknafXmrtk1/HxdSWso/vR5U/lQBoUU1ZFbgMM/WnUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFJQAjOsalmIVQMkngCuO17x2Iy0GnYZuhnYcfgP61c8Xabq2qNHDaAG125dd4XLZ71zP/CEav/zwT/v4tAFL4tzSSeHPD8jOS75ZmzyTtFeW+Y/95vzr2nx/4P1TXtD0a2s4VkmtgfMDOFx8oHfr0rhf+FTeJf8Anzj/AO/6/wCNAGxcs3/Clbc5OftPXP8A00NeceY/95vzr2KbwXqz/DKHRxAv29Zt5j8wYxvJ69OhrjP+FTeJf+fOP/v+v+NAHRfDFifBfiAkknd/7IKob29T+ddR4G8IanonhnV7O7hWOe4bMahwc/LjqKqf8IRq/wDzwT/v4tAF74fsTd32Tn9z/WuVZ23Hk9fWu78I+Hr7Sbi6e5jVBJHtXDA85rAbwTq+4/uE/wC/goAPBTE+IrfJJ+Vv5Gtb/hNJ9L1q7t7gefarMyj+8gz29aTwx4X1HTdYhuLiJViUMCQ4PUVV1bwhql3ql1NHCpjkkZlPmAcE0Ad1Y38GpW6zW8gkjPcdvY1YrhNA8P63pGoRSKipCzASjzAQV78V3VAC0UUUAFFFFABRRRQAUUUUAFFFFACE4GTwK8/8QeLJrzUkis5mit43A3IcFznk/Sug8YXd2tj9ls4ZZHm4do1J2r6fjXCw6LqAmjJspwNw/wCWZ9aAE+L2valpev2kdnfXFrG1sGKxSFQTubniuF/4THXf+gvef9/mruPi/ouoal4gtJLSyuLmNbYKWijLAHc3HFcL/wAInrX/AECrz/vw3+FAHS/GTxLq2k+C/CVxZaldWs86kyyQylWf5FPJHWvIP+FieKP+hg1L/wACW/xr134zeHtU1TwT4Rgs9OurqaFT5scMTMyfIo5AHFeP/wDCBeJf+gDqP/gM/wDhQB61d+KNYX9n+11MandDUGu9puvNPmEeYwxu69K8o/4WJ4o/6GDUv/Alv8a9VvPDeqt+z5a6cNNujfrd7jbeS3mAeYxztxnpXk//AAgXiX/oA6j/AOAz/wCFAHsfwm8S6tqnw58VXV3qV1c3MDHyppZSzJ8meD25rm/+Ex13/oL3n/f5q6L4S+HtU074c+Kra60+5triZj5cUsTKz/IBwCOa5v8A4RPWv+gVef8Afhv8KAPQPhFruo6pqWpJeX090qW4ZRLIWAOeozTG17Udx/0646/89DTvhDot/pupak13ZT2ytbhVMsZUE7ugzUbaJqG4/wChXHX/AJ5mgDa8I6te3WvQRzXU0sZDZVnJHSqet61fw6xeIl5MiLKwCq5AAzVvwhpd5ba9BJLazRoFbLMhA6VT1zR76bWL10tJnRpWIZYyQeaAE0fWr+XVbRHvJ2RpVBUucEZqn428Z6l4Z+IDNDcSPaxpGWtWY7GBXnjsferej6PfRatZu9nMqLKpLGMgDmsT4oaBqWoeMLma2sLieIxxgSRxFgfl9RQB7Bo+rW+uabBe2r74ZVyPUeoPuKu15H8K5tY0HUmsbvT7tLC5OdzxMBG/Y9OAen5V65QAUUUUAFFFFABRRRQAUlLVHUr77K1pGPvTzqn4dT/KgDzDxx8Rtb0PxPe2VpNElvGV2q0QY8qD1/GsW3+LXiOS4iRriHazgH9yvrVP4oKf+E41Lg9U/wDQFrm7NT9sg4P+sX+YoA7341fFDX/BPiSzs9KuIooJLUSsJIVc7tzDqfYCvPf+GgvGf/P5b/8AgMla/wC0spbxlp2AT/oK9v8AbavIvLb+6fyoA9u+Onxk8T+BfAfgzVNIuYYrzVEJuWkgVw37tW4B6ck14j/w1t8R/wDoI2f/AIBJXcftRKzfCv4bgKSfLPb/AKYpXzN5T/3G/KgD7GvPjV4qh/ZrtPGS3UA12S88lpfs67NvmsuNvToBXjX/AA1t8R/+gjZ/+ASV1epK3/DF+njac/2j0x/03evnPyn/ALjflQB9ofBn4xeJ/Gnwv8Ya3qlzDLqGmsRbukCoq/uw3IHXmuW/4aC8Z/8AP5b/APgMlU/2b1Zfgf8AEMEEHee3/TIVw3lt/dP5UAfRnwV+JmveNdT1WHVZ4pY7e2EsYjiVMNux2rLf4ueJFZgLmHAP/PBayv2alK63r2QR/oQ7f7VYUinzG4PU9qAPUvAPxD1rX/FFtZXk0b27q5ZViCnhSRzWf4k+KGv6br+oWsE8SwwzsiAwqSADxzWV8KFP/Cb2XH8En/oJrJ8ZKf8AhLNX4P8Ax9SfzoA6nw/8UfEGoa5p9rNcQmGadI3AhUHBODVD4tfFzxJ4R8bXWm6bcwx2kccbKrwKxyVyeTWF4SU/8JRpPB/4+o//AEIVlfH9WPxMvsAn9zD2/wBgUAKv7QHjIsB9st+v/PslfUtnI01pC7csyKT9SK+FFUh1yCOa+27G+8u5sbU9JLQOPqMf0P6UAa1FFFABRRRQAUUUUAFYmpa8bTXrKwWJX87G5m6rknp+VbdcLqchf4gW4/uPGv8A47n+tAFjXvFrabqs9uLKCUJj536ngGqUfjpmkRf7OtxkgVm+Mv8AkYrv/gP/AKCKyIP9fH/vD+dAHT/ELx4/hXVre2XTre7EkIk3zdRyRj9K5f8A4XHL/wBAOx/X/Cm/Gz/kZLP/AK9R/wChNXnlAHr3xG+JknhPw1oF+ulWt4b9cmKbO2P5QeOPevPP+Gip/wDoWNM/X/Cr3x0/5ELwX/uH/wBFrXiNAH0Z4m+Ms+h/BWLxcmi2csjXQh+wsSIhlyuenXivHP8Ahsu//wChN0f/AL6b/Ct34if8mm23/YQX/wBGtXytQB90fCn43XHjr4eeKNek0SysX0tiFt4Cdkvybvm4rnf+GkLv/oXNO/76P+Fct+zb/wAkO+If++f/AEUK4WgD6d+FHxWn8dX2pwyaVa2Itbfzg0BJLc4wa4l/2kr1XYf8I9YcHH32/wAKi/Zt/wCQx4g/68R/6FXkE3+tf/eNAH0T8N/jZdeMvF1rpUuj2lokquxliYlhtUn+lZnir4/3mheJNT05NDspltbh4hI7HLYOMniuJ+An/JTtO/65zf8AoBrn/iP/AMj94g/6/Zf/AEI0AeqeGv2grzWvEOm2DaFZQrdXCQmRGbK7mAyOK1fiZ8Zrjwb4uuNLj0azvFjjjfzpidx3LnHSvEvh/wD8jxoH/X9D/wChiun/AGgP+SmX3/XGH/0AUAby/tHXDMN3hrTyc/3j/hXt9/4i+w3mlxmBCt0qktnlM46fnXxan3l+tfWHjGQxnRnHVYAf5UAehUUUUAFFFFABRRRQAVzdxqWmJ4oW3ex3X29QLjA6lRg/lXSVw+rQmPx9aN2kZG/TH9KALGu6vo1rqk0d1ppnnXG6TA54HvVKPXvD7SIBpJByMHA/xrL8Zf8AIxXf/Af/AEEVkQf6+P8A3h/OgDqfiBr3h7S9Vgi1bSDqE7Qhlk2g4XJ45PrmuY/4THwT/wBCyf8Avhf8ai+Nn/IyWf8A16j/ANCavPKAPbfGOpeGV8O6JNqmii+s5FzbQlQfKG0e/pj8q47+2Ph5/wBCgv8A37X/ABq78Rv+RI8Kf9ch/wCgLXm1AHrviDUvB8fwrjuLzw6tz4e+0ALp2xcB95+bGcdcmvMP+Ei+EP8A0T2P/v0n/wAVXSeMP+Tf4f8Ar8X/ANGNXhFAH0x4E1rwX/wgfiO60bw0NO0mHJvLNUUef8noD6cVxf8AwsT4af8AQlN/37T/AOKo+F3/ACRzxz9G/wDRYrxugD6e+E/ijwlruoalHoGgHSZY7cNM5VRvTP3eCaw21j4ebmz4QUnPP7tf8awf2af+Q3r3/XkP/QqwpP8AWN9TQB6z4F1LwdceJLdNI8Orp98VfZcBFG0bTnofSuY8VeO/h/Y+JNTt7/wk11ex3DrNPsQ73B5b73eo/hP/AMjxZf7kn/oJry34j/8AI/eIP+v2X/0I0Aep+GvHnw9vPEOmwWXhFra8kuESGbYnyOWGG+92Na/xL8Y+CdH8W3FrrXhk6nqCxoWuNinIK8Dk9hXiPw//AOR40D/r+h/9DFdP+0B/yUy+/wCuMP8A6AKAN9fiJ8NNwx4KbP8A1zT/AOKr2bxBqmlWq2X2uwNwHi3R8D5V44618ap95frX1n4siM82iRgZLwqv54oA7+iiigAooooAKKKKACsnUrOxbVLG5uSy3AbZDtzgnrg1rVXvLVboQk9YpFkH4f8A1s0AcvrsPh5tUmN7NKtzxvC5x0GO3pVKO38K+Ym2efdkY+91/Ksvxl/yMV3/AMB/9BFZEH+vj/3h/OgDtPGmh+HNU1GGTV5JkuFi2qI2IG3J9B65rn/+ER8D/wDPe6/76b/CtL4hf8haD/riP5muWoA7rxFonh680XSoL+SZbOFcW5QnJG0Dnj0xXOf8Ij4H/wCe91/303+Fa3iz/kXdE/3P/ZRXJUAdbrWg+FZPh+ljeS3C6EJgwdWbfu3E9cZ65rgf+EJ+Ff8Az93/AP32/wD8TXUeK/8AklKf9fC/+hmvJaAPZfCvh/wdZ+DddtdMnuW0iYH7YzsxYfLg4yM9K4z/AIQn4V/8/d//AN9v/wDE1reBf+SceKfo3/oFeb0Aey/DPw/4O0m+v28OTXMs0kAWYTMxATPbIHeqreEfA+45nus5/vN/hWZ8E/8AkKar/wBew/8AQqjb7x+tAHW+FfDvhbT9bhn0yW4a8UNtEjMRgjnqPSuS8SeEfhvdeINQm1C5vFvpJ2adUd8ByecYX1rc8E/8jFb/AO638jXnHjL/AJGzV/8Ar6k/nQB1Ph/wj8NrXXdPmsLq9a9jnRoQzvguDxn5fWtT4g+GfAep+J57jXri6j1JkQOsTMFwBx0HpXn/AIS/5GjSf+vqP/0IVufFv/kdrr/rnH/6DQAo8FfCvI/0u/8A++3/APia9huLHTrjUNNDlzPFHugXnG0Y5P6V82L94fWvp+0tVaS3uTyywCMfjgn+lAF6iiigAooooAKKKKACiiigDjfEXh+wm1N7i71aKyeXBEchUcAY4yazovD+jLIhHiG2YhgQN6c/+PVs/ETwiPFWiMIlH263y8J9fVfx/nivAreNo76JHUq6yqCpHIOaAPfPFel6ffX0T3eqw2LiPASRlBIyeeTWJ/wjui/9DFa/99p/8VXNfGz/AJGOy/69R/6E1eeUAfQuuaXYXOk6dFPqkNtDGuI5mZQJOByMmsL/AIR3Rf8AoYrX/vtP/iq5/wCI3/IkeFP+uQ/9AWvN6APe9W8O6fqXgtbBtXijtPNDC8ypUkMTjrj9a43/AIVjof8A0Ndv/wCOf/FUlz/yRS3/AOvn/wBqGvN6APdPDfg+wsfC+r6fb6xHdw3WQ9wm3Efy47Gub/4VFpn/AEMcf/fK/wDxVO+GH/Ik+IP97/2QVQoA7TwL4JtPDF3dy2+qrfGaLYVUAbRnOeCagPg/TiT/AMTqL/x3/Gm/D7/j7vv+uP8AWuVb7x+tAHe+H/Ddlp+qRTw6nHcyKGAjXGTkexrm9b+F+nalrF7dSa8kDzSs7RlV+Uk9PvU/wT/yMVv/ALrf+gmqPiD/AJDl/wD9dm/nQBa0f4Xadp+rWd0mvpM8MqyLGFX5iDnH3qv+Mvh7Y+INelvZ9ZSzkZVUwlVOMDGeTWJof/IYsv8Arsv860fHH/IxTf7i/wAqAKEfwf06SRVTxCrsTwqopJ/8er1mGPyYUTOdqgZ+grjPAeg8/wBpTL7Qgj82rtqACiiigAooooAKKKKACiiigArzzxn8MDrWsRalpzxW8jOGnSTIDEH7wwOteh0UAeefEP4e6h4s1a3urSa3jSOERkSsQc5J7D3rlf8AhSutf8/Vl/303/xNaPxv+HeseIvJ1fRbiV5baLy5LKNypdQSdy4PJ56V86S6hqEMjJJdXSOpwytIwIPoRmgD6o8WeAb/AFzw7olhBNbpLZJtkaQnaflA449q5H/hSutf8/Vl/wB9N/8AE1ifF68uIfhr4FeOeVHaEbmVyCf3SdTXj39qXv8Az+XH/f1v8aAPqmbwDfyfD2LQhNB9qWbzC+TsxuJ9M965L/hSutf8/Vl/303/AMTWHeXlwP2cLSYTy+b9sx5m87v9a3evH/7Uvf8An8uP+/rf40AfWHg3wPe+H/D2qWFxNA8t0co0ZJA+XHPFVf8AhXt//wA9rf8A76P+FcJ8HLueb4Y+LnknkkdWO1mckj92Ohrlvt1z/wA/M3/fw/40Ae/+F/DFzos9y80kbiSPYNhPr9KxT8P78knzrf8A76P+Fc/8F7iabVNUEkryAWwI3MT/ABVG1zNuP72Tr/eNAHa+HfB93pOrRXMskLIoYEKTnkY9Kr6p4HvbzUbmdJYQkkhYBic4J+lZvguaR/ENuGkdhtbgsT2NUteuJV1q+AlcATNgBj60Ab2m+Br601C3naWErHIrEAnOAfpWlqnhF9W8QfapZFFrhdyjO44HSsTwpod9f3UN3JJJFbRsGBYn58dgPSvQqAEjjWGNURQqKMBR0Ap1FFABRRRQAUUUUAFFFFABRRRQAUUUUAFcH8QPg/o3jpXuCv2DVMcXcK/e/wB9f4v513lFAHi3xY+Hutap4P8AC2l6banUJ9PXy5mjYAcIozyR1xXlf/Cl/Gf/AEA5f+/if/FV9e1XurU3KYWaWA/3omxQB4xdeAtek+Bdtoa6e51VbrzGt9y5C+YxznOOhrzH/hS/jP8A6Acv/fxP/iq+qLzT7/8Asn7PbXzG6DZE8nBIz04Fc7Np/iyM/LctKPVZFH88UAch8L/A+uaD4B8S2F9YPb3d0xMMbMpL/JjsfWue/wCFa+JP+gXJ/wB9r/jXsmlw62uj3y3Rc3p/1OXUnp6g46+tZCWPi1jzK6e7Sp/SgDK+FfhPVdA1DUJL+za3SSAIhZgcnPTg1G3hLVtx/wBDbr/eH+NdvoNjq1s0rahdiYMuFUHO0+vSr1jpslr80t7cXLf7ZAH5CgDkPCvh/UNP1qGee2aOJQwLZHcfWtu38G27ancXt2fPMkhdYv4Rk9/WuhpaAEVQqgAYA4AFLRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAf/2Q==)

# In[86]:


df2['temp'].iloc[0] = np.NaN


# In[87]:


df2['temp'].fillna(df2['temp'].mean())


# In[88]:


# Check for correlation among numerical variables
num_var = ['pickups','spd','vsb','temp','dewp', 'slp','pcp01', 'pcp06', 'pcp24', 'sd']

corr = df[num_var].corr()


# In[89]:


corr


# In[97]:


# Check for correlation among numerical variables
num_var = ['pickups','spd','vsb','temp','dewp', 'slp','pcp01', 'pcp06', 'pcp24', 'sd']

corr = df[num_var].corr()

# plot the heatmap

plt.figure(figsize=(15, 7))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# - *As expected, temperature shows high correlation with dew point*
# - *Visibility is negatively correlated with precipitation. If it is raining heavily, then the visibility will be low. This is aligned with our intuitive understanding* 
# * *Snow depth of course would be negatively correlated with temperature.*
# * *Wind speed and sea level pressure are negatively correlated with temperature* 
# * *It is important to note that correlation does not imply causation*
# * *There does not seem to be a strong relationship between number of pickups and weather stats* 

# #### **Bivariate Scatter Plots**

# A **scatterplot** displays the relationship between 2 numeric variables. For each data point, the value of its first variable is represented on the X axis, the second on the Y axis
# 
# ![Screenshot 2022-02-26 153458.jpg](data:image/jpeg;base64,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)

# In[102]:


x = data.dtypes.reset_index()


# In[106]:


x[x[0] == "float64"]['index'].unique()


# In[107]:


num_cols = ['spd', 'vsb', 'temp', 'dewp', 'slp', 'pcp01', 'pcp06', 'pcp24',
       'sd']


# In[111]:


num_cols = data.select_dtypes('float').columns.tolist()


# In[184]:


sns.pairplot(data=df[num_var], diag_kind="kde")
plt.show()


# - *We get the same insights as from the correlation plot*
# - *There does not seem to be a strong relationship between number of pickups and weather stats*

# #### Now let's check the trend between pickups across different time based variables

# We can check the trends for time measures by plotting Line charts
# 
# A **line chart** is often used to visualize a trend in data over intervals of time, thus the line is often drawn chronologically.
# 
# ![Screenshot 2022-02-26 153349.jpg](data:image/jpeg;base64,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)

# #### **Pickups across months**

# In[114]:


cats


# In[115]:


df['month'] = df['month'].astype('object')


# In[118]:


df.columns


# In[120]:


df


# In[119]:


df[['start_month','pickups']]


# In[125]:


dir(df)


# In[124]:


cats = df.start_month.unique().tolist()
df.start_month = pd.Categorical(df.start_month, ordered=True, categories=cats)

plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="start_month", y="pickups", ci=False, color="red", estimator='sum',markers=True)
plt.ylabel('Total pickups')
plt.xlabel('Month')
plt.show()


# - *There is a clear increasing trend in monthly bookings*
# - *The number of pickups in June is almost 2.8 times of that of January*

# #### **Pickups vs Days of the Month**

# In[ ]:


plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="start_day", y="pickups", estimator='sum', ci=False, color="red")
plt.ylabel('Total pickups')
plt.xlabel('Day of Month')
plt.show()


# - *There is a steep fall in the number of pickups over the last days of the month*
# - *This can partially be attributed to month of Feb having just 28 days. We can drop Feb and have a look at this chart again*
# - *There is a peak in the bookings around the 20th day of the month*

# **Let us drop the observations for the month of Feb and see the trend**

# In[127]:


import numpy as np


# In[132]:


df[['start_day','start_month','start_hour','pickups']]


# In[138]:


months = df['start_month'].unique().tolist()


# In[143]:


req_months = list(set(months) - set(['February','March']))


# In[159]:


req_months


# In[167]:


df['borough'].unique()


# In[173]:


len(df[(df['borough'] == 'Bronx') | (df['borough'] == 'Brooklyn')])


# In[169]:


len(df[df['borough'].apply(lambda x: x in ('Bronx', 'Brooklyn'))])


# In[ ]:


df['start_month']


# In[164]:


len(df[df['borough'].apply(lambda x:x in ['Bronx'])])


# In[156]:


x = df[df['start_month'].apply(lambda x: x not in req_months)]


# In[158]:


x['start_month'].unique()


# In[128]:


# Let us drop the Feb month and see the trend
df_not_feb =  df[df['start_month'] != 'February']
plt.figure(figsize=(15,7))
sns.lineplot(data=df_not_feb, x="start_day", y="pickups", estimator=np.sum, ci=False, color="red")
plt.ylabel('Total pickups')
plt.xlabel('Day of Month')
plt.show()


# - *Number of pickups for 31st is still low because not all months have the 31st day*

# #### **Pickups across Hours of the Day**

# In[130]:


df[['start_hour','pickups']]


# In[ ]:


plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="start_hour", y="pickups", estimator='sum', ci=False, color="red")
plt.ylabel('Total pickups')
plt.xlabel('Hour of the day')
plt.show()


# - *Bookings peak around the 19th and 20th hour of the day*
# - *The peak can be attributed to the time when people leave their workplaces*
# - *From 5 AM onwards, we can see an increasing trend till 10, possibly the office rush*
# - *Pickups then go down from 10AM to 12PM post which they start increasing*

# #### **Pickups across Weekdays**

# In[ ]:


cats = ['Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']
df.week_day = pd.Categorical(df.week_day, ordered=True, categories=cats)

plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="week_day", y="pickups", ci=False, color="red", estimator='sum')
plt.ylabel('Total pickups')
plt.xlabel('Weeks')
plt.show()


# - *Pickups gradually increase as the week progresses and starts dropping down after Saturday*
# - *We need to do more investigation to understand why demand for Uber is low in the beginning of the week*

# **Let's check if there is any significant effect of the categorical variables on the number of pickups**

# #### **Pickups across Borough**

# In[177]:


plt.figure(figsize=(15,7))           
sns.boxplot(x=df['borough'], y=df['pickups'])
plt.ylabel('pickups')
plt.xlabel('Borough')
plt.show()


# In[174]:


c = 4


# In[176]:


print(id(c))


# In[178]:


len(df['borough'].unique())


# In[188]:


# Dispersion of pickups in every borough
sns.catplot(x='pickups', col='borough', data=df, col_wrap=4, kind="boxen")
plt.show()


# - *There is a clear difference in the number of riders across the different boroughs*
# - *Manhattan has the highest number of bookings*
# - *Brooklyn and Queens are distant followers*
# - *EWR, Unknown and Staten Island have very low number of bookings. The demand is so small that probably it can be covered by the drop-offs of the inbound trips from other areas*

# #### **Relationship between pickups and holidays**

# In[193]:


sns.catplot(x='hday', y='pickups', data=df, kind="bar", estimator=np.sum)
plt.show()


# - *The mean pickups on a holiday is lesser than that on a non-holiday*

# ### **Multivariate Analysis**

# In[199]:


sns.barplot(x='borough',y='pickups',data=df,hue='hday')
plt.xticks(rotation=90);


# In[197]:


sns.catplot(x='borough', y='pickups', data=df, kind="bar", hue='hday')
plt.xticks(rotation=90)
plt.show()


# The bars for EWR, Staten Island and Unknown are not visible. Let's check the mean pickups in all the borough to verify this.

# In[ ]:


# Check if the trend is similar across boroughs
df.groupby(by = ['borough','hday'])['pickups'].mean()


# - *In all the boroughs, except Manhattan, the mean pickups on a holiday is very similar to that on a non holiday*
# - *In Queens, mean pickups on a holiday is higher*
# - *There are hardly any pickups in EWR*

# Since we have seen that borough has a significant effect on the number of pickups, let's check if that effect is present across different hours of the day.

# In[ ]:


plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="start_hour", y="pickups", hue='borough', estimator='sum', ci=False)
plt.ylabel('Total pickups')
plt.xlabel('Hour of the day')
plt.show()


# - *The number of pickups in Manhattan is very high and dominant when we see the spread across boroughs*
# - *The hourly trend which we have observed earlier can be mainly attributed to the borough Manhattan, as rest of the other boroughs do not show any significant change for the number of pickups on the hourly basis*

# ### **Outlier Detection and Treatment**

# **Let's visualize all the outliers present in data together**

# In[ ]:


# outlier detection using boxplot
# selecting the numerical columns of data and adding their names in a list 
numeric_columns = ['pickups','spd','vsb','temp','dewp', 'slp','pcp01', 'pcp06', 'pcp24', 'sd']
plt.figure(figsize=(15, 12))

for i, variable in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    plt.boxplot(data[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# - *The `pickups` column has a wide range of values with lots of outliers. However we are not going to treat this column since the number of pickups can have a varying range and we can miss out on some genuine values if we treat this column*
# - *Starting from `spd` to `sd`, all the columns are related to weather. The weather related variables have some outliers, however all of them seem to be genuine values. So we are not going to treat the outliers present in these columns*

# ### **Actionable Insights and Recommendations**

# #### **Insights**
# 
# We analyzed a dataset of nearly 30K hourly Uber pickup informations, from New York boroughs.
# The data spanned over every day of the first six months of the year 2015.
# The main feature of interest here is the number of pickups. 
# Both from an environmental and business perspective, having cars roaming in an area while the demand is in another or filling the streets with cars during a low demand period while lacking during peak hours is inefficient. Thus we determined the factors that effect pickup and the nature of their effect.
# 
# We have been able to conclude that -  
# 
# 1. Uber cabs are most popular in the Manhattan area of New York
# 2. Contrary to intuition, weather conditions do not have much impact on the number of Uber pickups
# 3. The demand for Uber has been increasing steadily over the months (Jan to June)
# 4. The rate of pickups is higher on the weekends as compared to weekdays
# 5. It is encouraging to see that New Yorkers trust Uber taxi services when they step out to enjoy their evenings
# 6. We can also conclude that people use Uber for regular office commutes.The demand steadily increases from 6 AM to 10 AM, then declines a little and starts picking up till midnight. The demand peaks at 7-8 PM
# 7. We need to further investigate the low demand for Uber on Mondays
# 
# 
# #### **Recommendations to business**
# 
# 1. Manhattan is the most mature market for Uber. Brooklyn, Queens, and Bronx show potential 
# 2. There has been a gradual increase in Uber rides over the last few months and we need to keep up the momentum
# 3. The number of rides are high at peak office commute hours on weekdays and during late evenings on Saturdays. Cab availability must be ensured during these times
# 4. The demand for cabs is highest on Saturday nights. Cab availability must be ensured during this time of the week
# 5. Data should be procured for fleet size availability to get a better understanding of the demand-supply status and build a machine learning model to accurately predict pickups per hour, to optimize the cab fleet in respective areas
# 6. More data should be procured on price and a model can be built that can predict optimal pricing
# 7. It would be great if Uber provides rides to/from the JFK Airport, LaGuardia Airport airports. This would rival other services that provide rides to/from the airports throughout the USA. 
# 
# ####  **Further Analysis that can be done**
# 
# 1. Dig deeper to explore the variation of cab demand, during working days and non-working days. You can combine Weekends+Holidays to be non-working days and weekdays to be the working days
# 2. Drop the boroughs that have negligible pickups and then analyze the data to uncover more insights

# In[ ]:




