#!/usr/bin/env python
# coding: utf-8

# # EDA
# 

# In[1]:


# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# load the training and testing data files
train_df = pd.read_csv(r'D:\00RU\DS8015 - Machine learning\Project\train.csv')
test_df = pd.read_csv(r'D:\00RU\DS8015 - Machine learning\Project\test.csv')


# In[2]:


# check to see if the data files have any missing values
# RangeIndex: Gives dimension of training set
# Also tells us how many non-NA values for each feature
train_df.info()


# **Description of the features**
# 
# ---
# 
# PassengerId: Unique ID of a passenger
# 
# Survived:    0 - Not survived and 1 -  survived
# 
# Pclass:    Ticket class of passengers. It acts as a proxy for socio-economic status (SES). Pclass value is 1 for upper, 2 for middle and 3 for lower class.
# 
# Sex:    Sex     
# Age:    Age (in years). It is fractional if less than 1. If the age is estimated, is it in the form of xx.5    
# SibSp:    Number of siblings/spouse aboard the Titanic     
# Parch:    Number of parents / children aboard the Titanic     
# Ticket:    Ticket number     
# Fare:    Passenger fare     
# Cabin:    Cabin number     
# Embarked:   Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[3]:


train_df.describe(include= 'all')


# In[4]:


total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
missing_data.head(5)

# Cabin has high number of missing data hence dropping this feature is more logical than imputation 
# Age and Embarked still has acceptable number of missing entries and hence imputation can be performed here


# In[5]:


# We’ll need to fill the two missing values for Embarked. 
# Taking a quick look at the two passengers that don’t have values for Embarked
# Inner bracket gives boolean output and the outer train_df gives rows having null values as the output

print (train_df[train_df.Embarked.isnull()])


# Imputation in 'Embarked' feature - Approach 1

# In[6]:


# pivot table shows a breakdown by Sex, Pclass, Embarked, and shows the number of people from each subset that survived and embarked at a specific port
# This approach suggests that the imputation should be done with 'C' values as it is the most probable values for female passengers of class 1
print (train_df.pivot_table(values='Survived', index=['Sex', 'Pclass'], 
                     columns=['Embarked'], aggfunc='count'))


# Imputation in 'Embarked' feature - Approach 2

# In[7]:


bins = range(0,100,10)
df = train_df.copy()
df['Age1'] = pd.cut(df['Age'], bins)

#First filter the df of females who survived
#Create a pivot table on basis of age bins created before with column as embarkment
#This shows the missing data should be from S class 
# We will use 'S' for imputation as this approach is more logical and reliable


df1 = df[(df.Survived == 1) & (df.Sex == "female")]
print (df1.pivot_table(values='Survived', index=['Age1','Pclass'], 
                     columns=['Embarked'], aggfunc=['count']))


# In[8]:


# Finally, imputation of missing values by 'S'
(train_df.Embarked.iloc[61]) = 'S'
(train_df.Embarked.iloc[829]) = 'S'


# In[9]:


le_Sex = LabelEncoder()
train_df.Sex = le_Sex.fit_transform(train_df.Sex)
test_df.Sex = le_Sex.transform(test_df.Sex)

le_Embarked = LabelEncoder()
train_df.Embarked = le_Embarked.fit_transform(train_df.Embarked)
test_df.Embarked = le_Embarked.transform(test_df.Embarked)


# In[10]:


corrMatrix = train_df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
# Age is highly correlated with Pclass, hence we will use this feature to impute the missing Age values


# Dealing with missing entries of 'Age' feature

# In[11]:


# We can use the classical method such as imputation with mean 
# Another approach is generating a list of random numbers (with size = df['Age'].isnull() and values mean + std or mean - std) and filling NaN values with this list

# We can do even better by using the P_class feature!
# Pclass does not contain any missing data entries also we may see a relation of the passenger class with regards to the age of passengers
# As seen here the young people are more likely to travel in class 3 (cheapest)

sns.boxplot(x='Pclass',y='Age', data=train_df)
plt.show()


# In[12]:


train_df.groupby(by='Pclass').mean()['Age']
# These values can be imputed wherever the age is missing


# In[13]:


# Now, the aim is to fill in the Above values wherever age is missing. 
# Function can be made as shown here for this purpose

# ac = [['Age','Pclass']]
def impute_age(ac):
    Age = ac[0]
    Pclass = ac[0]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 25
    
    else:
        return Age


# In[14]:


data = [train_df,test_df]
for dataset in data:
  dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis=1).astype(int) 


# In[15]:


sns.heatmap(train_df.isnull())
# Now the only feature having missing values is Cabin but we are gonna drop this feature as it has 77% missing values!!
train_df = train_df.drop('Cabin',axis=1)
test_df = test_df.drop('Cabin',axis=1)


# In[16]:


sns.heatmap(train_df.isnull())


# Missing Values handled successfully!
# Now we will convert features

# In[17]:


test_df.info()


# In[18]:


# Convert Fare feature to integer and fill NaN with 0
data = [train_df,test_df]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[19]:


# Dropping name and ticket feature, as it is of less importance

train_df = train_df.drop(['Ticket','Name'],axis=1)
test_df = test_df.drop(['Ticket','Name'],axis=1)    


# In[20]:


#Looks good now
train_df.info()


# In[21]:


train_df.head(5)


# # FEATURE IMPORTANCE 

# In[22]:


#PCLASS
# Pclass is contributing to a persons chance of survival, especially if this person is in class 1. 

g = sns.catplot(x="Pclass", y="Survived", data=train_df, height=6,
                   kind="bar", palette="muted")

g.set_xticklabels(["1st", "2nd", "3rd"])
g.set_ylabels("survival probability")
plt.show()


# In[23]:


#SEX with Pclass
# Shows that in general females have higher probability of survival than men irrespective of the Pclass
g = sns.catplot(x="Sex", y="Survived", hue = "Pclass", data=train_df, height=6, 
                   kind="bar", palette="muted")

g.set_xticklabels(["Female", "Male"])
g.set_ylabels("survival probability")
plt.show()


# In[24]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
f = train_df[train_df['Sex']==0]
m = train_df[train_df['Sex']==1]
ax = sns.histplot(x="Age", hue="Survived",  data=f,ax = axes[0],bins=30, stat = 'percent')
ax.set_title('Female')
ax = sns.histplot(x="Age", hue="Survived",  data=m, ax = axes[1],bins=40, stat = 'percent')
ax.set_title('Male')
# 0 - Not survived; 1 - Survived
# Men have higher probability of survival for age group 18-35 years old
# Women have higher probability of survival for age group 16-34 years old (Ignoring the imputed valueof 25 years old)
# In general infants (<10 years) have high probability of survival irrespective of gender


# In[25]:


#SIBSP

g = sns.catplot(x="SibSp", y="Survived", data=train_df, height=6, 
                   kind="bar", palette="muted")
g = sns.catplot(x="Parch", y="Survived", data=train_df, height=6, 
                   kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
plt.show()


# In[26]:


#EMBARKED

g = sns.catplot(x="Embarked", y="Survived", hue='Sex', data=train_df, height=6, 
                   kind="bar", palette="muted")
g.despine(left=True)
g.set_xticklabels(['Cherbourg', 'Queensland', 'Southampton'])
g.set_ylabels("survival probability")
plt.show()


# In[27]:


# Scaling the Age feature
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 5, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# let's see how it's distributed 
train_df['Age'].value_counts()


# In[28]:


data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 10, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 10) & (dataset['Fare'] <= 20), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 35), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 35) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[29]:


train_df.head()


# # FEATURE ENGINEERING 

# In[30]:


# Function which checks if an individual has age less than 5 years and assigns it value 1 
# A new feature 'Child' is created in train and test data sets 

def is_child(x):
    if int(x) == 0 :
        return 1
    else:
        return 0

train_df['Child'] = train_df.Age.apply(is_child)
test_df['Child'] = test_df.Age.apply(is_child)


# In[31]:


g = sns.catplot(x="Child", y="Survived", data=train_df, height=6,kind="bar", palette="muted")
g.despine(left=True)
g.set_xticklabels(['Over 5yrs', 'Under 5yrs'])
g.set_ylabels("survival probability")
plt.show()

# Children have higher probability of survival


# In[32]:


# SIBSP & PARCH

train_df['FamSize'] = train_df.SibSp + train_df.Parch
test_df['FamSize'] = test_df.SibSp + test_df.Parch


# In[33]:


g = sns.catplot(x="FamSize", y="Survived", data=train_df, height=6, kind="bar", palette="muted")
g.set_ylabels("survival probability")
plt.show()
# This shows that the survival probability for an individual with less or equal 3 relatives are high


# In[34]:


# Creating new feature based on family size


def is_alone(x):
    if int(x) == 0:
        return 1
    else:
        return 0

train_df['Alone'] = train_df.FamSize.apply(is_alone)
test_df['Alone'] = test_df.FamSize.apply(is_alone)

def is_small_fam(x):
    if int(x) == 0:
        return 0
    elif int(x) < 4:
        return 1
    else:
        return 0

train_df['SmallFam'] = train_df.FamSize.apply(is_small_fam)
test_df['SmallFam'] = test_df.FamSize.apply(is_small_fam)

def is_large_fam(x):
    if int(x) >= 4:
        return 1
    else:
        return 0

train_df['LargeFam'] = train_df.FamSize.apply(is_large_fam)
test_df['LargeFam'] = test_df.FamSize.apply(is_large_fam)


# In[67]:


train_df['Fare_Per_Person']=train_df['Fare']/(train_df['FamSize']+1)


# In[69]:


train_df.head()


# In[36]:


# In this section I am confirming whether there is presence of multicolinearity!
# Using Variance Inflation Factor as shown here 

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_variables = train_df.drop('Survived',axis=1)
vif_data = pd.DataFrame()

vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]

vif_data
#Variance Inflation Factor of all the features is less than 10 so there is no chance of the multicolinearity amongst the independent variables


# In[37]:


g = sns.catplot(x="SmallFam", y="Survived", data=train_df, height=6, 
                   kind="bar", palette="muted")
g.set_xticklabels(['No', 'Yes'])
g.set_ylabels("survival probability")
plt.show()
g = sns.catplot(x="LargeFam", y="Survived", data=train_df, height=6, 
                   kind="bar", palette="muted")
g.set_xticklabels(['No', 'Yes'])
g.set_ylabels("survival probability")
plt.show()


# In[38]:


# FARE 

def is_low_fare(x):
    if int(x) == 0 or int(x) == 1:
        return 1
    else:
        return 0

train_df['LowFare'] = train_df.Fare.apply(is_low_fare)
test_df['LowFare'] = test_df.Fare.apply(is_low_fare)


# In[39]:


(train_df['LowFare'] == 1)


# In[40]:


g = sns.catplot(x="LowFare", y="Survived", data=train_df, height=6,kind="bar", palette="muted")
g.despine(left=True)
g.set_xticklabels(['Over 20 pounds', 'Under 20 pounds'])
g.set_ylabels("survival probability")
plt.show()
#Higher the fare better the passenger class and hence higher the probability of survival


# # CLASSIFICATION MODEL

# In[41]:


# BUILDING THE MODEL

#from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# we don't need the PassengerIds anymore and we don't want them to influence the
# classifier, we will need the PassengerIds for the training set later on when 
# we write the output file with all the the predictions


# In[42]:


X = train_df.drop(['Survived'], axis=1)
Y = train_df.Survived

x_train, x_test, y_train, y_test = train_test_split(X, Y) # default split is 75% train and 25% test


# In[ ]:


fixed_params = {'objective':'binary:logistic',
                'scale_pos_weight':1.605
                }

xgb = XGBClassifier(**fixed_params)


# In[ ]:


test_params = {'n_estimators':np.array([25, 50, 100, 150, 200]),
               'learning_rate':np.logspace(-4, -1, 4),
               'max_depth':np.array([3, 4, 5, 6, 7]),
               'gamma':np.array([0.0, 0.1]),
               'max_delta_step':np.array([0.0, 0.001, 0.01]),
               'reg_lambda':np.array([0.01, 0.1])
               }


# In[46]:


my_score = 'accuracy'
#grid_search = GridSearchCV(xgb, test_params, n_jobs=-1, verbose=1,
                           #scoring=my_score, cv=3)
#grid_search.fit(x_train, y_train)


# # MODEL EVALUATION

# In[61]:


print ('Best %s score: %0.3f'%(my_score, grid_search.best_score_))
print ('Best Parameters:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(test_params.keys()):
    print ('\t%s: %r'%(param_name, best_parameters[param_name]))


# In[ ]:


predictions_test = grid_search.predict(x_test)

print ('Scores for final validation set:')
print ('\taccuracy score: %f'%accuracy_score(y_test, predictions_test))
print ('\tprecision score: %f'%precision_score(y_test, predictions_test))
print ('\trecall score: %f'%recall_score(y_test, predictions_test))
print ('\tf1 score: %f'%f1_score(y_test, predictions_test))


# # Random forest

# In[62]:


# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[63]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.copy()


# In[65]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(y_test, Y_pred)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# In[51]:


#KNN
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(X_test)  
acc_knn = round(knn.score(Y_test, Y_pred) * 100, 2)


# In[52]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(Y_test, Y_pred) * 100, 2)


# In[53]:


sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(Y_test, Y_pred)

acc_sgd = round(sgd.score(Y_test, Y_pred) * 100, 2)


# In[54]:


gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train)  
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(Y_test, Y_pred) * 100, 2)


# In[55]:


perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(Y_test, Y_pred) * 100, 2)


# In[56]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(Y_test, Y_pred) * 100, 2)


# In[59]:


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN','Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# # K folds cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[ ]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)


# In[ ]:


importances.plot.bar()


# In[ ]:


train_df  = train_df.drop("Child", axis=1)
test_df  = test_df.drop("Child", axis=1)

train_df  = train_df.drop("Alone", axis=1)
test_df  = test_df.drop("Alone", axis=1)


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# In[ ]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# In[ ]:


from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
print(confusion_matrix(Y_train, predictions))

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))

