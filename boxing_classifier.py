import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

def init(df):
    #to_drop=['judge1_A', 'judge1_B', 'judge2_A', 'judge2_B', 'judge3_A','judge3_B']
    #df.drop(to_drop,inplace=True,axis=1)
    df['result'] = df['result'].map({'draw': 0, 'win_A': 1,'win_B':2})
    df['stance_A']=df['stance_A'].map({'orthodox':0,'southpaw':1})
    df['stance_B']=df['stance_B'].map({'orthodox':0,'southpaw':1})
    df['decision']=df['decision'].map({'SD':1,'MD':2,'UD':3,'KO':4,'TKO':5,'DQ':6,'RTD':7})
    return df

def removeNan(df):
    df.fillna(-1,inplace=True)
    return df

def removeRows(df,t):
    df.dropna(thresh=t,inplace=True)
    return df

def removeColumns(df,x):
    df.drop(x,inplace=True,axis=1)
    return df

def normalize(df):
    x=df.values
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(x)
    df=pd.DataFrame(x_scaled,columns=df.columns)
    return df

def generate(df,name):
    df.to_csv(name,encoding='utf-8',index=False,sep=',')

def separate(df):
    features = ['age_A', 'age_B', 'height_A', 'height_B', 'reach_A', 'reach_B',\
       'stance_A', 'stance_B', 'weight_A', 'weight_B', 'won_A', 'won_B',\
       'lost_A', 'lost_B', 'drawn_A', 'drawn_B', 'kos_A', 'kos_B',\
       'decision', 'judge1_A', 'judge1_B', 'judge2_A', 'judge2_B', 'judge3_A',\
       'judge3_B']
    dx = df.loc[:,features].values
    dy = df.loc[:,['result']].values
    x = pd.DataFrame(dx,columns=features)
    y = pd.DataFrame(dy,columns=['result'])
    return x,y

def split(x,y):
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=0)
    return xtr,xte,ytr,yte
    

def pca(df1,df2,n):
    pca = PCA(n)
    pca.fit(df1)
    x = pca.transform(df1)
    y = pca.transform(df2)
    return x,y

def runner(x,y):
   x,y = separate(x)
   x = removeNan(x)
   y = removeNan(y)
   xtr,xte,ytr,yte = split(x,y)
   xtr = normalize(xtr)
   xte = normalize(xte)
   xtr,xte = pca(xtr,xte,.95)
   return xtr,xte,ytr,yte

def c(x1,x2,y1,y2):
    clf = mlp(hidden_layer_sizes=(13,13,13),solver='sgd',\
                       learning_rate_init=0.01,max_iter=500)
    clf.fit(x1,y1.values.ravel())
    predictions = clf.predict(x2)
    print(confusion_matrix(y2.values.ravel(),predictions,labels=[0,1,2]))
    print(classification_report(y2,predictions,target_names=['draw','win_A','win_B']))
    
#textName = input("Type filename here : ")
df=pd.read_csv("usethis.csv")
x = y = xtr = xte = ytr = yte = df.copy()
xtr,xte,ytr,yte = runner(x,y)
c(xtr,xte,ytr,yte)
