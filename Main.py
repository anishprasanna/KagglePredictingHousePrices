#Anish Prasanna
#PipeLine/Feature Manipulation repurposed from https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#converts csv to DF


def convertToDF(file):
    df = pd.read_csv(file)
    return df

# train and test sets of the Data
sampSub = convertToDF('house-prices-advanced-regression-techniques/sample_submission.csv')
train = convertToDF('house-prices-advanced-regression-techniques/train.csv')
test = convertToDF('house-prices-advanced-regression-techniques/test.csv')
y = train.pop('SalePrice').values # just the training labels
trainfeatures = train.iloc[:, :-1]

#Encoding string column
HS = train[['HouseStyle']].copy() #One column DF
#print(HS)
#print(train['HouseStyle'].value_counts()) # variable categories
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
hs_train_trans = ohe.fit_transform(HS) # encodes each of the 8 values as a binary column
#print(hs_train_trans)
feature_names = ohe.get_feature_names() # column names of the variable now encoded
#print(feature_names)

#verification of encoding
firstrow = hs_train_trans[0]
#print(feature_names[firstrow == 1])
#print(HS.values[0])

#transform to original data

hs_inv = ohe.inverse_transform(hs_train_trans)
#print(hs_inv)
#print(np.array_equal(hs_inv,HS.values))

#transforming test set
test = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
hs_test = test[['HouseStyle']].copy()
hs_test_transformed = ohe.transform(hs_test)
#print(hs_test_transformed)
#print(hs_test_transformed.shape)

si_step = ('si', SimpleImputer(strategy = 'constant',fill_value='MISSING')) #imputes missing training data
ohe_step = ('ohe', OneHotEncoder(sparse=False,handle_unknown='ignore'))
steps = [si_step,ohe_step]
pipe = Pipeline(steps)
#for training set
hs_train = train[['HouseStyle']].copy()
hs_train.iloc[0,0]=np.nan
hs_transformed = pipe.fit_transform(hs_train)
#for test set
hs_test = test[['HouseStyle']].copy()
hs_test_transformed = pipe.transform(hs_test)

#encoding multiple strings
stringcols = ['RoofMatl','HouseStyle']
string_train = train[stringcols]
string_train_transformed = pipe.fit_transform(string_train)

#ColumnTransformer allows for specific transformations for specified columns
cat_si_step = ('si', SimpleImputer(strategy='constant',fill_value='MISSING'))
cat_ohe_step = ('ohe', OneHotEncoder(sparse=False,handle_unknown='ignore'))
cat_steps = [cat_si_step, cat_ohe_step]
cat_pipe = Pipeline(cat_steps)
cat_cols = ['RoofMatl', 'HouseStyle']
cat_transformers = [('cat', cat_pipe, cat_cols)]
ct = ColumnTransformer(transformers=cat_transformers)

#categories that were transformed
X_cat_transformed = ct.fit_transform(train)
#transform test set
X_cat_transformed_test = ct.transform(test)

#feature names
pl = ct.named_transformers_['cat']
ohe = pl.named_steps['ohe']
#print(ohe.get_feature_names())

#Pipelining numeric columns
#print(train.dtypes)
kinds = np.array([dt.kind for dt in train.dtypes])
#print data types of columns
print('Column Types -------')
print(kinds)
all_columns  = train.columns.values
is_num = kinds != 'O'
#numeric columns
num_cols = all_columns[is_num]
print("Numeric Columns -------")
print(num_cols)
#category columns
cat_cols = all_columns[~is_num]
print('Categorical Columns --------')
print(cat_cols)

#standardizing numeric columns
#imputes for missing vals
num_si_step = ('si', SimpleImputer(strategy='median'))
#scales columns
num_ss_step = ('ss',StandardScaler())
num_steps = [num_si_step,num_ss_step]

num_pipe = Pipeline(num_steps)
num_transformers = [('num',num_pipe,num_cols)]

ct = ColumnTransformer(transformers=num_transformers)
X_num_transformed = ct.fit_transform(train)

#combining categorical and numerical transformations
transformers = [('cat', cat_pipe, cat_cols),('num', num_pipe, num_cols)]
ct = ColumnTransformer(transformers=transformers)
#final transformed training set
X = ct.fit_transform(train)
ml_pipe = Pipeline([('transform', ct),('Ridge',Ridge())])
ids = test.iloc[:,0]
ml_pipe.fit(train, y)
#print(ml_pipe.score(train, y))
#cross validation
kf  = KFold(n_splits=5,shuffle = True, random_state=1)
#print(cross_val_score(ml_pipe,train,y,cv=kf))
#print(cross_val_score(LinearRegression(),X,y,cv=kf))

#CrossValScore for Ridge
print("Ridge CrossValScore: ", cross_val_score(ml_pipe,train,y,cv=kf).mean())
print("ElasticNet CrossValScore: ",cross_val_score(ElasticNet(),X,y,cv = kf).mean())
print("DecisionTreeRegressor CrossValScore: ",cross_val_score(DecisionTreeRegressor(),X,y,cv = kf).mean())

i = 1
KNNeigh = [1,2,3,4,5,6,7,8,9,10]
KNNvalues = []
while i<=10:
    print("K-Nearest Neighbors CrossValScore: ",i,' ',cross_val_score(KNeighborsRegressor(n_neighbors=i),X,y,cv = kf).mean())
    KNNvalues.append(cross_val_score(KNeighborsRegressor(n_neighbors=i),X,y,cv = kf).mean())
    i+=1

#print(ml_pipe.score(train,y))
#Parameter Search for Ridge
param_gridRidge = {
    'transform__num__si__strategy': ['mean', 'median'],
    'Ridge__alpha': [.001, 0.1, 1.0, 5, 10, 50, 100, 1000]}
gsRidge = GridSearchCV(ml_pipe, param_gridRidge
                  , cv=kf)
gsRidge.fit(train,y)
pd.DataFrame(gsRidge.cv_results_).to_excel('GridSearchRidge.xlsx')
#print('Ridge Output done')
#Parameter Search for KNN
param_gridKNN = {
        'n_neighbors': [1, 2,3, 4, 5, 6, 7, 8,9,10]}
gsKNN = GridSearchCV(KNeighborsRegressor(), param_gridKNN
                  , cv=kf)
gsKNN.fit(X,y)
pd.DataFrame(gsKNN.cv_results_).to_excel('GridSearchKNN.xlsx')
#print("Lasso: ",cross_val_score(Lasso(),X,y,cv = kf).mean())
#print('KNN Output done')
#Parameter Seach for GradientBoostingRegression
param_gridGrad = {
    'n_estimators': [500], 'max_depth': [4], 'min_samples_split': [2],
    'learning_rate': [.1], 'loss': ['ls']
}
gsGrad = GridSearchCV(GradientBoostingRegressor(), param_gridGrad
                  , cv=kf)
gsGrad.fit(X,y)
pd.DataFrame(gsGrad.cv_results_).to_excel('GridSearchGradientBoosting.xlsx')
print('Gradient Output done')
print('Best Scores for Each Alg: ')
print('Best Ridge Score: ',gsRidge.best_score_)
print('Best KNN Score: ',gsKNN.best_score_)
print('Best GradientBoosting Score: ',gsGrad.best_score_)

#Comparison Graph
performance =  [gsRidge.best_score_, gsKNN.best_score_,  gsGrad.best_score_]
objects = ('Ridge','KNN','Gradient Boosting')
ypos = np.arange(len(objects))
plt.bar(ypos,performance,align='center',alpha=.5)
plt.xticks()
plt.xticks(ypos, objects)
plt.ylabel('R^2 Score')
plt.title('Optimized Algorithms Score ')
plt.savefig('AlgScoreComp.png')
plt.clf()
plt.plot(KNNeigh,KNNvalues)
plt.savefig('KNN CrossValScores')


#KNN Neighbors
#SubmissiontoCSV
ml_pipe = Pipeline([('transform', ct),('Gradient',GradientBoostingRegressor(n_estimators= 500, max_depth= 4, min_samples_split= 2,
learning_rate= .1, loss= 'ls'))])
ids = test.iloc[:,0]
ml_pipe.fit(train, y)
predictions  = ml_pipe.predict(test)
output = pd.DataFrame({'Id': ids, 'SalePrice': predictions})
output.to_csv('housepredictionsGradientBoosting.csv',index=False)