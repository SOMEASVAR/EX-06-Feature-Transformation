# EX-06-Feature-Transformation

# AIM:
To Perform the various feature transformation techniques on a-Picture and save the data to a file. 

# EXPLANATIION:
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM:
## STEP 1:
Read the given Data.
## STEP 2:
Clean the Data Set using Data Cleaning Process.
## STEP 3:
Apply Feature Transformation techniques to all the feature of the data set.
## STEP 4:
Save the data to the file.


# CODE:
```

Developed By: R.SOMEASVAR
Register No: 212221230103

```
# titanic-Picture.csv
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic-Picture.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  

from sklearn.preprocessing import OrdinalEncoder  
 
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  

df  
```

## FUNCTION TRANSFORMATION: 
``` 
#Log Transformation  
np.log(df["Fare"])  
#ReciprocalTransformation  
np.reciprocal(df["Age"])  
#Squareroot Transformation:  
np.sqrt(df["Embarked"]) 
``` 

## POWER TRANSFORMATION:  
```
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  
df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df    
df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df  
df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  
df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df
```  

## QUANTILE TRANSFORMATION:
```
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)  
df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show()  
sm.qqplot(df['Age_1'],line='45')  
plt.show()  
df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  
sm.qqplot(df['Fare_1'],line='45')  
plt.show()  

df.skew()  
df 
```
# OUTPUT:
## Reading the data set:
![output](./Titanic-Pictures/o1.png)
## Cleaning the-Picture:
![output](./Titanic-Pictures/o2.png)
![output](./Titanic-Pictures/o3.png)
![output](./Titanic-Pictures/o4.png)
## FUNCTION TRANSFORMATION:
![output](./Titanic-Pictures/o6.png)
![output](./Titanic-Pictures/o7.png)
## POWER TRANSFORMATION:
![output](./Titanic-Pictures/o8.png)
![output](./Titanic-Pictures/o9.png)
![output](./Titanic-Pictures/o10.png)
![output](./Titanic-Pictures/o11.png)
![output](./Titanic-Pictures/o12.png)
## QUANTILE TRANSFORMATION
![output](./Titanic-Pictures/o13.png)
![output](./Titanic-Pictures/o14.png)
![output](./Titanic-Pictures/o15.png)
![output](./Titanic-Pictures/o16.png)
## Final Result:
![output](./Titanic-Pictures/o17.png)
![output](./Titanic-Pictures/o19.png)

# data_to_transform.csv
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  
df=pd.read_csv("Data_To_Transform.csv")  
df  
df.skew()  
```

## FUNCTION TRANSFORMATION:  
```
#Log Transformation  
np.log(df["Highly Positive Skew"])  
#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"])  
#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"])  
#Square Transformation  
np.square(df["Highly Negative Skew"])  
```
## POWER TRANSFORMATION:  
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])  
df  
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])  
df  
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])  
df  
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])  
df 
``` 

## QUANTILE TRANSFORMATION:  
```
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal')  
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()  
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()  
df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 
df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show()  

df.skew()  
df 
```


# Output:
## Reading the data set:
![output](./Transform-Picture/s1.png)
![output](./Transform-Picture/s2.png)
## FUNCTION TRANSFORMATION:
![output](./Transform-Picture/s3.png)
![output](./Transform-Picture/s4.png)
![output](./Transform-Picture/s5.png)
![output](./Transform-Picture/s6.png)
## POWER TRANSFORMATION:
![output](./Transform-Picture/s7.png)
![output](./Transform-Picture/s8.png)
![output](./Transform-Picture/s9.png)
![output](./Transform-Picture/s10.png)
## QUANTILE TRANSFORAMATION:
![output](./Transform-Picture/s12.png)
![output](./Transform-Picture/s13.png)
![output](./Transform-Picture/s14.png)
![output](./Transform-Picture/s15.png)
![output](./Transform-Picture/s17.png)
![output](./Transform-Picture/s18.png)
![output](./Transform-Picture/s19.png)
## Final Result:
![output](./Transform-Picture/s20.png)
![output](./Transform-Picture/s21.png)


# RESULT:
Hence, Feature transformation techniques is been performed on given-Picture and saved into a file successfully.