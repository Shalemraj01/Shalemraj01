#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')


# In[3]:


data = pd.read_csv(r"C:\Users\shale\Downloads\car-mpg.csv")
data


# In[4]:


data.head()


# In[17]:


car_data = data.drop('car_name',axis=1)
car_data


# In[18]:


car_data['origin'] = car_data['origin'].replace({1:'america',2:'europe',3:'asia'})
car_data


# In[19]:


#convert categorical variable into dummy/indicator variables. As many columns will be created as distinct values
#This is also known as one hot encoding. The column names will be America,Europe and Asia... with one hot encoding

car_data = pd.get_dummies(car_data,columns= ['origin'])
car_data


# In[20]:


#lets analyze the distribution of the dependent(mpg)column
car_data.describe().transpose()


# In[21]:


sns.pairplot(data = car_data,diag_kind ='kde')


# In[22]:


#hp column is missing in the describe it means there's something wrong with that column


# In[23]:


#check if hp column contains anything other than digits
#run the isdigit() check on hp column of the car_data dataframe. Result will be true or false for every row
#capture the result in temp and dow a frequency count using value_counts()
#


# In[24]:


temp = pd.DataFrame(car_data.hp.str.isdigit()) #if the string is made of digits store True else False for every row
temp[temp['hp']== False] #from temp take only those rows where hp is false


# In[25]:


#on inspecting records num 32,126 etc we find '?' in the columns. Replace them with "nan"
#replace them with nan and remove the records from data frame that have nan


# In[26]:


car_data = car_data.replace('?',np.nan)
car_data


# In[27]:


#let us see if we can get those records with nan


# In[28]:


car_data[car_data.isnull().any(axis=1)]


# In[29]:


#there are varios ways to handle missing values. Drop the rows, replace missing values with median values;


# In[30]:


car_data.median()


# In[31]:


car_data = car_data.apply(lambda x:x.fillna (x.median()),axis=0)
car_data


# In[32]:


car_data.dtypes


# In[33]:


car_data['hp'] = car_data['hp'].astype(float)  #converting the hp column from object/string to float
car_data


# In[34]:


car_data.describe()


# In[35]:


#let us to correlation analysis among the different dimensions and also each dimension with dependent dimension
#this is done using scatter matrix function which creates a dashboard reflecting useful information about the dimensions


# In[36]:


car_data_attr = car_data.iloc[:,0:10]
car_data_attr


# In[37]:


#let us do a correlation analysis among the different dimensions and also each dimension with the dependent dimension
sns.pairplot(car_data_attr,diag_kind = 'kde') #to plot the density curve insted of the histogram


# In[38]:


x = car_data.drop('mpg',axis=1)
x
x = x.drop({'origin_america','origin_asia','origin_europe'},axis=1)


y = car_data[['mpg']]uir


# In[39]:


x


# In[40]:


#let us break the x and y dataframes into training set and test set for this we will use
#sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split


# In[41]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state=1)


# In[42]:


from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model.fit(x_train,y_train)


# In[43]:


#let us explain the coefficients for each of the independent attributes

for idx,col_name in enumerate(x_train.columns):
    print("The coefficients for {} is {}".format(col_name,regression_model.coef_[0][idx]))


# In[44]:


intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))


# In[45]:


#model score - R2 or coeff of determinant
#R^2 = 1-RSS/TSS = RegErr/TSS
regression_model.score(x_train,y_train)


# In[46]:


#----------------Using statsmodel library to get R type outputs


# In[47]:


#R^2 is not a reliable metric as it always increases with increase of variables even if the attributes have no influence 
#on the predicted variable. Instead we use adjusted R^2 which removes the statistical chance that improves R^2 
#scikit doesnot provide a facility for adjusted R^2... so we use
#statsmodel, a library that gives results similar to
#what you obtain in R language
#This library expects X and Y to be given in single dataframe


data_train = pd.concat([x_train,y_train],axis=1)
data_train


# In[48]:


import statsmodels.formula.api as smf
lm1 = smf.ols(formula = 'mpg ~ cyl+disp++hp+wt+acc+yr+car_type', data= data_train).fit()
lm1.params


# In[49]:


print(lm1.summary())


# In[50]:


#let us check the sum of squared errors by predicting the value of y for test cases and
#substracting the actual y for the test series

mse = np.mean((regression_model.predict(x_test)-y_test**2))
mse


# In[ ]:





# In[ ]:




