## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
### Developed by : Ashwin Kumar A
### Reg No : 212223040021
```python
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/83d6e9bf-3ecf-4a9d-90ea-49b0a53c8a67)

Ordinal Encoding
```py
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
pm=["Hot","Warm","Cold"]
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/88d12df0-3759-4be3-bf2b-a350ca6d7579)

```py
df["bo2"]=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/2ad3adb4-fe3a-4585-9369-04ed587e6492)
Label Encoder
```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/828d5103-7120-442e-8462-837c77145eb7)
One Hot Encoding
```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/d5e096f7-b080-426c-8178-b599d7edb397)

```py
 pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/6a649ee5-b271-45d9-b2eb-6281d38e2176)

```py
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/ff631ce6-e747-4176-9b8b-bc837d916ee3)

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/f734ef85-50aa-4151-9a7a-ce93a60d34b5)

```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/a5c349dc-9f90-41ad-9a27-57255b409477)

MEAN ENCODEING
```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/7ef1e0db-4897-4dbd-a7f6-3cbee5973c10)
FeatureTransformation
```py
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/db8bba92-2fd3-4d4b-bf69-355a9ca507d4)

```py
df.skew()
```

![image](https://github.com/user-attachments/assets/4ff25818-46a2-4d89-86b1-6ad833d7fc3b)
1.LOG transformation
```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/e13669e8-04d7-4fb9-82e8-2b62fb3f4792)
2.Reciprocal Transformation
```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1d27f513-e25b-4e05-bef2-79909ee25758)

```py
 np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d5059cc6-148b-4933-943f-4355820495c9)
```py
 np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/9076ea13-d8f0-4221-9616-54726e03eecb)
Power Transformations BOX COX
```py
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
![image](https://github.com/user-attachments/assets/863d5385-5bf3-4531-8fc6-eab9bf59fc60)

```py
df.skew()
```
![image](https://github.com/user-attachments/assets/e8eff2da-8b99-4b73-a736-815386ae6cb0)

```py
 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()
```
![image](https://github.com/user-attachments/assets/7ac2eca0-e42b-46ca-b8ca-0c6b7ae194cd)
Quantile transformation
```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/e485ae00-8e53-4f22-9228-9475456f1274)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/6d10b410-38e9-45ea-a6df-b1aaaaa4c1ee)

```py
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/60c9883f-5ef9-451d-8dca-f8c8239fb270)

```py
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/ab3a5549-fa24-4409-bf97-742c4221e1b7)

```py
 dt=pd.read_csv("titanic_dataset.csv")
 dt
```
![image](https://github.com/user-attachments/assets/42c89e8f-210a-424a-b308-3c70e3937653)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a3b788b7-38cc-463c-b402-fb00028bb084)

```py
df["Highly Negative Skew_1"], parameters = stats.yeojohnson(df["Highly Negative Skew"])
sm.qqplot(df["Highly Negative Skew_1"], line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/beec3874-9bae-49a4-b16b-8cc54832b99c)


# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.       
