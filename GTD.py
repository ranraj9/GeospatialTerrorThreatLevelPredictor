
# coding: utf-8

# ## Read Data

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
pd.set_option('display.max_columns', 500)


# In[2]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb


# In[3]:


df=pd.read_csv("data.csv",encoding='latin1')
df.head()


# In[4]:


df.shape


# ## Data Cleaning

# ### Focussing on India and its neighbouring countries

# In[5]:


df['region_txt'].value_counts()


# In[7]:


df=df[df['region_txt']=='South Asia']


# In[8]:


df.head()


# In[9]:


df['country_txt'].value_counts()


# In[12]:


l3=['Maldives','Bhutan','Mauritius','Afghanistan']
for i in range(len(l3)):
    df = df.drop(df[df.country_txt == l3[i]].index)


# In[13]:


df['country_txt'].value_counts()


# ### Handling Missing Values

# In[14]:


missing=df.isnull().sum()*100/len(df)
missing


# In[15]:


missing50=missing[missing>50]


# In[16]:


missing50


# In[17]:


missing50.index[0]


# In[18]:


l1=[missing50.index[i] for i in range(len(missing50))]


# In[19]:


type(l1)


# In[20]:


df1=df.drop(labels=l1,axis=1)
df1.shape


# In[24]:


df2=df1
df2.head()


# In[25]:


z=df2['provstate'].value_counts()
z[:30]


# In[26]:


l4=[];
for i in set(df2['provstate']):
    if z[i]<100:
        #print(i)
        l4.append(i)
len(l4)


# In[27]:


for i in range(len(l4)):
    df2 = df2.drop(df2[df2.provstate == l4[i]].index)
df2['provstate'].value_counts()


# In[28]:


df2['targtype1_txt'].value_counts()


# In[29]:


miss=df2.isnull().sum()*100/len(df2)
miss


# In[30]:


df2.shape


# In[31]:


df2.dropna(subset=['corp1', 'targsubtype1','latitude','longitude'],inplace=True)
df2.shape


# In[32]:


df2['weapsubtype1_txt'] = df2['weapsubtype1_txt'].fillna('Unknown Gun Type')


# In[33]:


df2['ishostkid'] = df2['ishostkid'].fillna(0)


# In[34]:


df2['natlty1_txt'] = df2['natlty1_txt'].fillna('Others')


# In[35]:


df2.dropna(subset=['guncertain1'],inplace=True)
df2.shape


# In[36]:


df2['claimed'].value_counts()


# In[37]:


df2['claimed'] = df2['claimed'].fillna(0)


# In[38]:


df2['nkill'].describe()


# In[39]:


df2['nkill'] = df2['nkill'].fillna(0)


# In[40]:


df2['nwound'].describe()


# In[41]:


df2['nwound'] = df2['nwound'].fillna(0)
df2['nwoundte'] = df2['nwoundte'].fillna(0)
df2['nkillter'] = df2['nkillter'].fillna(0)


# In[42]:


df2['nperps'].describe()


# In[43]:


def make_others(s):
    if s == -99.0:
        return -1
    else:
        return s
df2['nperps'] = df2['nperps'].apply(make_others)


# In[44]:


df2['nperps'] = df2['nperps'].fillna(-1)


# In[45]:


df2['nperpcap'].describe()


# In[46]:


def make_others(s):
    if s == -99.0:
        return -1
    else:
        return s
df2['nperpcap'] = df2['nperpcap'].apply(make_others)


# In[47]:


df2['nperpcap'] = df2['nperpcap'].fillna(0)


# In[48]:


df2.isnull().sum()*100/len(df2)


# ### Feature Engineering

# #### Threat Level Of a Group : no. of incidents, no. people killed and wounded, propety damage, no. of perpetrator killed or wounded

# In[49]:


df2.gname.value_counts()


# In[50]:


grouped=df2.groupby('gname')


# In[51]:


df3=grouped.agg({'eventid':{'No. of incidents':'count'},
             'nkill': 'sum',
             'nwound': 'sum',
             'nkillter': 'sum',
             'nwoundte': 'sum',
             'property': 'count',
             'ishostkid': 'count',
             'iyear': {
                 'recency':lambda x: max(x)-1970,
                 'frequency': lambda x: max(x)-min(x)
             }
            })
df3.head(10)


# In[52]:


df3.sort_values(("eventid",'No. of incidents'),ascending=False, inplace=True)
df3.head()


# In[53]:


sns.distplot(df3['eventid']['No. of incidents'], fit=norm);
fig = plt.figure()


# In[54]:


df3['totalcount'] = np.log1p(df3['eventid']['No. of incidents'])


# In[55]:


sns.distplot(df3['totalcount'], fit=norm);
fig = plt.figure()


# In[56]:


sns.distplot(df3['nkill']['sum'], fit=norm);
fig = plt.figure()


# In[57]:


df3['logkill'] = np.log1p(df3['nkill']['sum'])
sns.distplot(df3['logkill'], fit=norm);
fig = plt.figure()


# In[58]:


df3['logwound'] = np.log1p(df3['nwound']['sum'])
sns.distplot(df3['logwound'], fit=norm);
fig = plt.figure()


# In[59]:


df3['logkillter'] = np.log1p(df3['nkillter']['sum'])
sns.distplot(df3['logkillter'], fit=norm);
fig = plt.figure()


# In[60]:


df3['logwoundter'] = np.log1p(df3['nwoundte']['sum'])
sns.distplot(df3['logwoundter'], fit=norm);
fig = plt.figure()


# In[61]:


sns.distplot(df3['property']['count'],fit=norm);
fig = plt.figure()


# In[62]:


df3['logproperty'] = np.log1p(df3['property']['count'])
sns.distplot(df3['logproperty'], fit=norm);
fig = plt.figure()


# In[63]:


sns.distplot(df3['iyear']['frequency'],fit=norm);
fig = plt.figure()


# In[64]:


df3['logfrequency'] = np.log1p(df3['iyear']['frequency'])
sns.distplot(df3['logfrequency'], fit=norm);
fig = plt.figure()


# In[65]:


df3['recency'] = (df3['iyear']['recency'])
sns.distplot(df3['recency'], fit=norm);
fig = plt.figure()


# In[66]:


df3.head(3)


# In[67]:


df3.drop([('eventid','No. of incidents'),('nkill','sum'),('nkillter','sum'),('nwound','sum'),('nwoundte','sum'),('property','count'),('ishostkid','count'),('iyear','recency'),('iyear','frequency')],axis=1,inplace=True)
df3.head()


# In[68]:


df3['threat']=df3['totalcount']+df3['logkill']*2+df3['logwound']*2+df3['logproperty']*0.5+df3['logfrequency']*0.5+df3['recency']*0.1
df3.head(3)


# In[69]:


df99=df3[['threat']]
df4=set(df2['gname'])
df4 = pd.DataFrame({'gname':list(df4)})
df4=pd.merge(df4,df99,left_on='gname',right_index=True)
df4.head()


# In[70]:


df4['Threat']=df4[('threat', '')]
df4.drop([('threat', '')],axis=1,inplace=True)
df4.head()


# In[71]:


sns.distplot(df4['Threat'], fit=norm);
fig = plt.figure()


# In[72]:


df2=pd.merge(df2,df4,on='gname',how='left')


# In[73]:


df2.shape


# In[74]:


df2.head(3)


# In[75]:


df2.drop(['region', 'region_txt'], axis=1, inplace=True)


# ### Categorical variables handle

# In[82]:


#Too many categories in the following columns
#df2['targsubtype1_txt'].value_counts() others karo less than 40
#df2['corp1'].value_counts() others karo
#df2['natlty1_txt'].value_counts() <20
#df2['gname'].value_counts() <10, <5


# In[76]:


z=df2['targsubtype1_txt'].value_counts()
sns.countplot(x=df2['targsubtype1_txt'])
len(z)


# In[77]:


others = []
for i,v in z.items():
    if v<40:
        others.append(i)
len(others)


# In[78]:


for index, row in tqdm(df2.iterrows()):
    if df2.loc[index,'targsubtype1_txt'] in others:
        df2.loc[index,'targsubtype1_txt'] = 'Others'
df2.shape


# In[79]:


z=df2['corp1'].value_counts()


# In[80]:


def make_cat(s):
    s = s.lower()
    if 'police' in s:
        return 'Police'
    elif 'navy' in s or 'army' in s or 'armed' in s or 'military' in s:
        return 'Army'
    elif 'force' in s:
        return 'Force'
    elif 'gov' in s or 'govt' in s or 'government' in s or 'ministry' in s:
        return 'Government'
    elif 'civillian' in s:
        return 'Civillian'
    elif 'enforcement' in s:
        return 'Enforcement'
    elif 'united nation' in s:
        return 'UN'
    elif 'teachers' in s or 'school' in s or 'education' in s or 'university' in s:
        return 'Education'
    elif 'organisation' in s or 'organization' in s or 'org' in s:
        return 'Org'
    elif 'party' in s or 'national' in s or 'parliament' in s:
        return 'National'
    elif 'mosque' in s or 'temple' in s or 'church' in s or 'gurudwara' in s or 'religion' in s:
        return 'Religion'
    elif 'rail' in s or 'air' in s or 'bus' in s or 'road' in s:
        return 'Transport'
    elif 'new' in s:
        return 'Media'
    elif 'hospital' in s or 'medical' in s:
        return 'Health'
    elif 'plant' in s:
        return 'Power'
    elif 'border' in s:
        return 'Border'
    else:
        return 'Others'


# In[81]:


df2['org'] = df2['corp1'].apply(make_cat)


# In[82]:


df2['org'].value_counts()


# In[83]:


sns.countplot(x=df2['org'])


# In[84]:


z=df2['natlty1_txt'].value_counts()
z


# In[85]:


others = []
for i,v in z.items():
    if v<10:
        others.append(i)
len(others)


# In[86]:


for index, row in tqdm(df2.iterrows()):
    if df2.loc[index,'natlty1_txt'] in others:
        df2.loc[index,'natlty1_txt'] = 'Others'


# In[87]:


sns.countplot(x=df2['natlty1_txt'])


# In[88]:


z=df2['gname'].value_counts()
z


# In[89]:


others = []
for i,v in z.items():
    if v<10:
        others.append(i)
len(others)


# In[90]:


def make_others(s):
    if s in others:
        return 'Others'
    else:
        return s
df2['gname'] = df2['gname'].apply(make_others)


# In[91]:


df2['gname'].value_counts()


# In[92]:


sns.countplot(x=df2['gname'])


# In[93]:


df2.head(3)


# In[94]:


# dfa=df2[['eventid','iyear','imonth',
      #'iday','country_txt','provstate','city','latitude','longitude','gname','targtype1_txt','Threat']]


# In[95]:


# dfa.to_csv('dash.csv', sep=',')


# In[96]:


dropped=['summary','weapdetail','dbsource','scite1','corp1','city','eventid','country','latitude','longitude','nkill','nkillus','nkillter','nwound','nwoundus','nwoundte','target1','attacktype1','targtype1','targsubtype1','natlty1','nperpcap','weaptype1','weapsubtype1']
df2.drop(dropped,axis=1,inplace=True)

# In[98]:


df2.drop(['scite2'],axis=1,inplace=True)


# In[99]:


df2.head(3)


# In[102]:


df2.shape


# In[103]:


df2.dtypes


# In[104]:


#df2.to_csv('clean_data_35_notencoded.csv', sep=',')


# In[105]:


# c=['org','targsubtype1_txt','country_txt','provstate','attacktype1_txt','targtype1_txt','natlty1_txt','gname','weaptype1_txt','weapsubtype1_txt']
# le=LabelEncoder()
# for i in c:
    # df2[i] = df2[i].astype('category')
    # df2[i] = le.fit_transform(df2[i])

# df2 = pd.read_csv('clean_data_35_notencoded.csv')

c = df2.columns
col = [i for i in c if i != 'iyear' and i != 'imonth' and i != 'iday' and i != 'Threat' and i != 'provstate' and i != 'nperps']
len(col)

df2.columns

df3 = df2[col]

df4 = pd.get_dummies(df3)

df5 = df2[['iyear', 'imonth', 'iday', 'Threat', 'nperps']]

df6 = pd.concat([df5, df4], axis = 1)

MONTHS = 12
DAYS = 30

df6['sine_month'] = np.sin(2*np.pi*df6.imonth/MONTHS)
df6['cos_month'] = np.cos(2*np.pi*df6.imonth/MONTHS)

df6.drop('imonth', axis=1, inplace=True)

df6['sine_day'] = np.sin(2*np.pi*df6.iday/DAYS)
df6['cos_day'] = np.cos(2*np.pi*df6.iday/DAYS)

df6.drop('iday', axis=1, inplace=True)


# In[106]:




# In[107]:


df6.reset_index(inplace = True)

# In[108]:

df6.drop(['index'],axis=1,inplace=True)
df6.head()
len(df6.columns)

# In[109]:


df6.shape

sc = StandardScaler()

names = ['iyear', 'Threat', 'nperps']

df7 = df6.copy()
df6.head()

df7[['iyear', 'Threat', 'nperps']] = sc.fit_transform(df7[['iyear', 'Threat', 'nperps']])


# In[110]:


df7.to_csv('encoded.csv', sep=',')

# ## MODEL


# In[110]:


Y=df2.iloc[:, 5].values


# In[111]:


len(set(Y))


# In[112]:


df2.drop(['provstate'],axis=1,inplace=True)


# In[113]:


df2.head(3)


# In[114]:


X = df2.values


# In[115]:


df2.shape


# In[116]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[117]:




# In[118]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[113]:


# lg = lgb.LGBMClassifier(silent=False)
# param_dist = {"objective" : "multiclass",
#               'boosting_type': 'gbdt',
#               'num_class':99,
#               "max_depth": [5,20, 40],
#               "learning_rate" : [0.001,0.01,0.1],
#               "num_leaves": [30,600,900],
#               "n_estimators": [200,300],
#               'metric': 'multi_logloss',
#              }


# In[115]:


# grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
# grid_search.fit(x_train,y_train)


# In[ ]:


# grid_search.best_estimator_


# In[119]:


d_train = lgb.Dataset(x_train, label=y_train)
params = {"objective" : "multiclass",'boosting_type': 'gbdt','num_class':35,"max_depth": 15,
          "learning_rate" : 0.001,"num_leaves": 2000,'metric': 'multi_logloss'}


# In[120]:


cat =[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31]


# In[ ]:


model2 = lgb.train(params, d_train,700, categorical_feature = cat)


# In[125]:


ypred2=model2.predict(x_test)


# In[126]:


ypred1=np.argmax(ypred2,axis=1)


# In[128]:


y_test[0:5]


# In[129]:


ypred1[0:5]


# In[132]:


accuracy_score(ypred1,y_test)


# In[133]:


y_train_p=model2.predict(x_train)


# In[134]:


y_train_p=np.argmax(y_train_p,axis=1)


# In[135]:


accuracy_score(y_train_p,y_train)
