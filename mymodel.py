import pandas as pd
import numpy as np

df=pd.read_csv('insurance.csv')
# separating columns into catcol and numcol
catcol = []
numcol = []
for i in df.dtypes.index:
  if df.dtypes[i] == "object":
    catcol.append(i)
  else:
    numcol.append(i)
# removing Outliers
from scipy.stats import zscore
features=df[['age', 'bmi', 'children', 'expenses']]
z=np.abs(zscore(features))
newdf=df[(z<=3).all(axis=1)]

# removing skewness
skew1 = ['age', 'bmi', 'children', 'expenses']
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer(method ="yeo-johnson")

# feature encoding
from sklearn.preprocessing import OrdinalEncoder
oe =OrdinalEncoder()
newdf.loc[:, catcol] = oe.fit_transform(newdf[catcol])

# feature scaling
x=newdf.drop("expenses", axis =1)
y=newdf["expenses"]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# model development
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=40)
from xgboost import XGBRFRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def mymodel(model):
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    train=model.score(xtrain,ytrain)
    test=model.score(xtest,ytest)

    #print(f"Traning accuracy:{train}\nTesting accuracy:{test}\n\n")
    mae=mean_absolute_error(ytest,ypred)
    #print("MAE :",mae)
    mse=mean_squared_error(ytest,ypred)
    #print("MSE :",mse)
    rmse=np.sqrt(mse)
    #print("RMSE :",rmse)
    r2=r2_score(ytest,ypred)
    #print("R2_Score",r2)
    avg_rmse=np.mean(rmse)
    #print("Average RMSE:", avg_rmse)
    return model

xg = mymodel(XGBRFRegressor(n_estimators=500,max_depth=4,learning_rate=1,colsample_bynode=0.8))
#print(cross_val_score(xg,xtrain,ytrain,cv=5).mean())

def mymodel(model):
    model.fit(xtrain,ytrain)
    return model
def makepredict():
    xg=XGBRFRegressor()
    model=mymodel(xg)
    return model