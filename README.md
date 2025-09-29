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
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="450" height="451" alt="image" src="https://github.com/user-attachments/assets/f3468f80-af28-451b-95c4-7a2eb1183a33" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="283" height="243" alt="image" src="https://github.com/user-attachments/assets/625d3374-59c1-4303-8f78-cc2ba231f140" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="488" height="446" alt="image" src="https://github.com/user-attachments/assets/029901bd-ce15-4230-abbc-cd98caf69362" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="478" height="449" alt="image" src="https://github.com/user-attachments/assets/80b74ca5-4985-439c-8016-91c7f1ddfca5" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(handle_unknown='ignore')
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]).astype(int))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="938" height="458" alt="image" src="https://github.com/user-attachments/assets/2494df80-e2c4-47db-9e69-d0db6931309d" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="1184" height="452" alt="image" src="https://github.com/user-attachments/assets/944803a5-bf7c-411a-b036-9aa1d096e150" />

```
pip install --upgrade category_encoders

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

<img width="734" height="510" alt="image" src="https://github.com/user-attachments/assets/ba2a134d-58a5-4ecc-b510-cff3e03e2819" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="732" height="520" alt="image" src="https://github.com/user-attachments/assets/743f060b-f1ed-419d-8615-722cf418d5c3" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="973" height="515" alt="image" src="https://github.com/user-attachments/assets/afd57ed1-b177-4c04-ad46-8ff76a839708" />


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="827" height="463" alt="image" src="https://github.com/user-attachments/assets/6472a056-465d-4ff6-9290-f9739101f760" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="1104" height="534" alt="image" src="https://github.com/user-attachments/assets/b07f1ef8-dc54-41cd-a6f6-e23d2b67e68f" />

```
df.skew()
```
<img width="498" height="266" alt="image" src="https://github.com/user-attachments/assets/bdc2a70a-13fe-4223-8a5b-dafb8e4523a6" />

```
np.log(df["Highly Positive Skew"])
```

<img width="470" height="537" alt="image" src="https://github.com/user-attachments/assets/9831e4e9-78ab-49ce-a74e-b52dd9b65bb9" />

```
np.reciprocal(df["Moderate Positive Skew"])

```

<img width="413" height="517" alt="image" src="https://github.com/user-attachments/assets/ea5fe141-8576-48a5-8cb2-3d1e6351a654" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="472" height="533" alt="image" src="https://github.com/user-attachments/assets/58043ef9-6457-4083-8d52-21cea3fdff79" />

```
np.square(df["Highly Positive Skew"])
```

<img width="538" height="528" alt="image" src="https://github.com/user-attachments/assets/00ff8317-8716-4c75-ad0d-96dee67e4855" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1378" height="535" alt="image" src="https://github.com/user-attachments/assets/80cddee1-cda9-4345-9d12-dcc503aa8e51" />

```
df.skew()
```

<img width="533" height="265" alt="image" src="https://github.com/user-attachments/assets/b24d05ba-dca4-4ba9-9451-e85b8a828434" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="622" height="301" alt="image" src="https://github.com/user-attachments/assets/56b1c266-95f9-418e-99fe-4c0dca97d7e7" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1770" height="550" alt="image" src="https://github.com/user-attachments/assets/84b97ee6-72c0-4b52-93bc-35c8dfa1b295" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="953" height="556" alt="image" src="https://github.com/user-attachments/assets/de6cc13d-d659-4364-beec-04fe7d18cc2f" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="836" height="549" alt="image" src="https://github.com/user-attachments/assets/635d3c7d-7762-4760-87dc-b5f4a56b52e1" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="860" height="558" alt="image" src="https://github.com/user-attachments/assets/0eb11312-9b3c-40c0-96ed-e3128ba56885" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="822" height="554" alt="image" src="https://github.com/user-attachments/assets/ad97407b-5ffd-4585-9709-412f6c82edcc" />





# RESULT:
      Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
