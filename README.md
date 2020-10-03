# Predicting Hospital Admissions using Claims Data


## Summary

With the rise of readily available data, health-centers nationwide are actively working to minimize costs while producing optimal health outcomes. The only question remains is as follows: how do we predict costs? Hospitalization time is a key driver in healthcare billing. In the **Heritage Health Data Challenge** via Kaggle, I sought to address this via basic classification model building. In addition to exploratory data analysis and feature engineering, I fit three models. The **random forest** algorithm was the most accurate, yielding an accuracy rate of 86.7%

## Introduction

Health informatics can carry significant impact with regards to costs and availability of services. The Heritage Health Competition was a past data competition hosted on Kaggle. Participants use available patient data to predict which patients are more likely to experience readmission. 

In this project, I use the past datasets to conduct data cleaning, exploratory data analysis, modeling, and appropriate predictive analysis.



## Data Processing 

The datasets were released via Kaggle in CSV formats. They contain many instances of incomplete cases and require extensive cleaning. The tables were pulled from a relational database, in which the member id is the primary field linking tables. Therefore, joins are required; the **members** and **target** tables have one-to-one relationships, they can be merged using left and/or inner joins. The **drugs** and **labs** tables have a one-to-many relationship with the member table, as they contain records on a yearly basis.


###  Feature Engineering

One aspect of this project, which may differ from how other participants approached the challenge, entails my experience as a hospital volunteer, a public health student, and later a research assistant. Based on this, rather than employing forward or backward stepwise model building, I will be deliberately selecting features that have documented impacts on health. 

One feature that I will be constructing is an SES categorical variable (`low_SES`), derived from the pay delay field. Pay delays can be the result of financial hardship, as I've learned through first hand experience. Socioeconomic status is a key determinant of health and will therefore be included in model building. 

Another feature I will be adding is the count of timepoints within a year (`time_count`) in which a patient has a claim. So if a patient has a claim at 0-1 months and 3-4 months during Year One, this feature would be a value of 2.



## Exploratory Data Analysis

The first step in any data-based problem is understanding the features and outcome we're working with. In addition to visualizing frequencies of specific demographic categories and clinical variables, we'll also visualize the days of hospitalization outcome variable (`length_recoded`).




![png](img/output_37_0.png)



![png](img/output_38_0.png)



![png](img/output_39_0.png)




![png](img/output_40_0.png)




![png](img/output_41_0.png)


### Outcome Visualized



![png](output_43_1.png)



![png](output_44_0.png)


## Modeling

Since we're looking at multiple outcomes with a series of co-variates, decision tree learning would be most appropriate in this scenario. This is further supported knowing that the winning methods used extended decision tree modeling for their predictive analyses.


### Linear Regression: Low SES

While `low_SES` was an engineered feature, based on the patient's pay delay, I was curious what it's impact was on other features. This was a question of my own (external to the challenge). I fit an OLS Linear Regression Model. The `low_SES` coefficient, when controlling for age categories, was statistically significantly associated (β = 2.6505, p < 0.0001) with the total number of conditions attributed toward hospitalization (per patient). Nothing definitive can be concluded from this, but it is still an interesting observation altogether. 


                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                    sum   R-squared (uncentered):                   0.725
    Model:                            OLS   Adj. R-squared (uncentered):              0.725
    Method:                 Least Squares   F-statistic:                          4.672e+04
    Date:                Fri, 02 Oct 2020   Prob (F-statistic):                        0.00
    Time:                        21:02:31   Log-Likelihood:                     -3.3906e+05
    No. Observations:              141558   AIC:                                  6.781e+05
    Df Residuals:                  141550   BIC:                                  6.782e+05
    Df Model:                           8                                                  
    Covariance Type:            nonrobust                                                  
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    AgeAtFirstClaim_20-29     2.5657      0.032     79.290      0.000       2.502       2.629
    AgeAtFirstClaim_30-39     2.7627      0.025    111.896      0.000       2.714       2.811
    AgeAtFirstClaim_40-49     3.0008      0.021    145.983      0.000       2.960       3.041
    AgeAtFirstClaim_50-59     3.3048      0.021    154.550      0.000       3.263       3.347
    AgeAtFirstClaim_60-69     3.8142      0.019    199.888      0.000       3.777       3.852
    AgeAtFirstClaim_70-79     4.2291      0.017    253.426      0.000       4.196       4.262
    AgeAtFirstClaim_80+       4.5627      0.023    197.753      0.000       4.517       4.608
    low_SES                   2.6505      0.015    172.416      0.000       2.620       2.681
    ==============================================================================
    Omnibus:                     9157.372   Durbin-Watson:                   1.433
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            11675.912
    Skew:                           0.612   Prob(JB):                         0.00
    Kurtosis:                       3.693   Cond. No.                         2.80
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


### Classification Models: Decision Tree

Note that this tree was fit without any dimmensionality reduction. As a result, there's definitely room for pruning and making the model more parsimonious. While the tree is ridiculously large and not as helpful as we'd like, the feature importance is worth noting: besides the time variables, `RESPR4` (*acute respiratory infections*), `ARTHSPIN` (*arthropathies and spine disorders*), `NEUMENT`(*neurological problems*), and `low_SES` were ranked the most important features. Overall, the model was 77.4% accurate.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>DSFS</td>
      <td>0.188571</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Year</td>
      <td>0.062379</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARTHSPIN</td>
      <td>0.037378</td>
    </tr>
    <tr>
      <th>44</th>
      <td>low_SES</td>
      <td>0.037099</td>
    </tr>
    <tr>
      <th>36</th>
      <td>RESPR4</td>
      <td>0.035005</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NEUMENT</td>
      <td>0.034198</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MISCHRT</td>
      <td>0.031744</td>
    </tr>
    <tr>
      <th>18</th>
      <td>INFEC4</td>
      <td>0.030325</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MSC2a3</td>
      <td>0.027085</td>
    </tr>
    <tr>
      <th>40</th>
      <td>SKNAUT</td>
      <td>0.026664</td>
    </tr>
  </tbody>
</table>




```python
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import auc
from sklearn import metrics
pred = tree_clf.predict(X_test) 
print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred)))
```

    Accuracy: 0.774


### Classification Models: Random Forest

For this model, I utilized a grid search to optimize parameters based on accuracy and refit accordingly. The model was 86.7% accurate. Furthermore, one of the key advantages of random forest was being able to visualize feature importance. `GIBLEED` and `ROAMI` were the leading clinical features. 


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
rnd_clf = RandomForestClassifier(n_estimators=200, max_leaf_nodes=16, n_jobs=-1, random_state = 11) 
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```


```python
param_grid = { "n_estimators": [100, 200, 250],
              "max_depth": [1, 5, 7, 11], 
              "criterion": ["gini", "entropy"] }
```


```python
model = model_selection.GridSearchCV( estimator=rnd_clf, param_grid=param_grid, 
                                     scoring="accuracy", 
                                     verbose=10, 
                                     n_jobs=1, cv=2 )
model.fit(X_train, y_train)
print(f"Best score: {model.best_score_}")
print("Best parameters set:", model.best_estimator_.get_params())

#Based on this grid search, we can use max_depth = 1 and n_estimators = 100
```


```python
rnd_clf = RandomForestClassifier(n_estimators=100, max_depth = 1, n_jobs=-1, random_state = 11) 
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred_rf)))
```

    Accuracy: 0.867



```python
feat = np.array(['name', 'score'])
for name, score in zip(X.columns.values, rnd_clf.feature_importances_): 
    z = np.array([name, score])
    feat = np.vstack((feat, z))

feature_tbl = pd.DataFrame(feat[1:,], columns = feat[0])
feature_tbl = feature_tbl.sort_values(by = ["score"], ascending = False)
feature_tbl.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>GIBLEED</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>45</th>
      <td>DSFS</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>37</th>
      <td>ROAMI</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>42</th>
      <td>TRAUMA</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ODaBNCA</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>



