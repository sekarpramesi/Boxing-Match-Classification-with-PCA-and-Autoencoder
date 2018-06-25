import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as mlx
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def init(df):
    #to_drop=['judge1_A', 'judge1_B', 'judge2_A', 'judge2_B', 'judge3_A','judge3_B']
    #df.drop(to_drop,inplace=True,axis=1)
    df['stance_A']=df['stance_A'].map({'orthodox':0,'southpaw':1})
    df['stance_B']=df['stance_B'].map({'orthodox':0,'southpaw':1})
    return df

def removeNan(df,x):
    df.fillna(x,inplace=True)
    res = df.values
    return df

def removeRows(df,t):
    df.dropna(thresh=t,inplace=True)
    return df

def removeColumns(df,x):
    df.drop(x,inplace=True,axis=1)
    return df

def normalize(x):
    scaler=preprocessing.StandardScaler()
    x_scaled=scaler.fit_transform(x)
    return x_scaled

def generate(df,name):
    df.to_csv(name,encoding='utf-8',index=False,sep=',')

def separate(df):
    features = ['age_A', 'age_B', 'height_A', 'height_B', 'reach_A', 'reach_B',\
       'stance_A', 'stance_B', 'weight_A', 'weight_B', 'won_A', 'won_B',\
       'lost_A', 'lost_B', 'drawn_A', 'drawn_B', 'kos_A', 'kos_B',\
       'judge1_A', 'judge1_B', 'judge2_A', 'judge2_B', 'judge3_A',\
       'judge3_B']
    t1= ['result']
    t2=['decision']
    dx = df.loc[:,features].values
    dy1 = df.loc[:,t1].values
    dy2 = df.loc[:,t2].values
    x = pd.DataFrame(dx,columns=features)
    y1 = pd.DataFrame(dy1,columns=t1)
    y2 = pd.DataFrame(dy2,columns=t2)
    y1= y1.iloc[:,0].str.replace(' ','').str.get_dummies(sep=',')
    y2= y2.iloc[:,0].str.replace(' ','').str.get_dummies(sep=',')
    print(y1.head(5))
    print(y2.head(5))
    y  = pd.concat([y1,y2], axis=1)
    print(y.head(5))
    return x,y

def split(x,y):
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.1,random_state=0)
    return xtr,xte,ytr,yte
    

def pca(df1,df2,n):
    pca = PCA(n)
    pca.fit(df1)
    x1 = pca.transform(df1)
    x2 = pca.transform(df2)
    return x1,x2

def runner(x):
   x,y = separate(x)
   x = removeNan(x,-1)
   y = removeNan(y,0)
   xtr,xte,ytr,yte = split(x,y)
   xtr = xtr.values
   xte = xte.values
   ytr = ytr.values
   yte = yte.values
   xtr = normalize(xtr)
   xte = normalize(xte)
   xtr,xte = pca(xtr,xte,.95)
   return xtr,xte,ytr,yte

def classify(x1,x2,y1,y2):
    clf = mlx(solver='adam', alpha=1e-5,\
                   hidden_layer_sizes=(13,13,13), random_state=1,activation='relu')
    clf.fit(x1,y1)
    predictions = clf.predict(x2)
    cols=['draw','win_A','win_B','DQ','KO','MD','NWS','PTS','RTD','SD','TD','TKO','UD'];
    #print(f1_score(y2,predictions,labels=cols,average='weighted'))
    #print(accuracy_score(y2,predictions, normalize=False, sample_weight=None))
    n_classes=y1.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y2[:, i],
                                                            predictions[:, i])
        average_precision[i] = average_precision_score(y2[:, i], predictions[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y2.ravel(),
        predictions.ravel())
    average_precision["micro"] = average_precision_score(y2, predictions,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    res = pd.DataFrame(predictions,columns=cols)
    return res

df=pd.read_csv("boxing.csv")
df=init(df)
x = y = xtr = xte = ytr = yte = df.copy()
xtr,xte,ytr,yte = runner(x)
res = classify(xtr,xte,ytr,yte)
#print(res.loc[res['draw'] == 1].count)
#print(res.loc[res['win_A'] == 1].count)
#print(res.loc[res['win_B'] == 1].count)
