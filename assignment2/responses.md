# Question 1
### Rounded Error Rates Q1

| SVM Linear | SVM RBF | RF | KNN1 | KNN5 | KNN10 |
|:--------|:-----:|:-----|:-----:|:--------|:-----:|
| 0.028 | 0.009 | 0.015 | 0.011 | 0.012 | 0.014 |


## A
Support Vector Machine with RBF kernel performs the best in this machine learning problem with the lowest 0.009 error rate, All the KNN's and RF have slightly more rates than SVM rbf.
## B

Support Vector Machine uses a hyperplane (boundary) to differentiate the two classes. The kernel fuction is used to transform data into higher dimensional data. The RBF kernel has a boundary which are not straight lines (radial) and should be used in non-linear datasets like our's to get the best result, the linear kernel has the boundary in the form of a straight line. if the dataset is not linear in nature or mixed with each other linear kernel will perform bad or fail, that's what happened here in Q1. KNNs at lower value of K has lesser error rates but are not reliable enough because it always tends to favour the dominant class. Also, Random Forest mostly performs better with higher volume of data.

## C
In the first assignment for lower values of K (0-7) the error rate was almost 0, this was because the data was distributed in a liner and consecutive manner leaving no scope for overlapping, for K = 10  the error rate was 0.002 which is pretty low again due to less distortion or overlapping in data and. In comparison the error rates in this question is 0.011, 0.012, 0.014 for K = 1, 5 and 10 respectively, which are almost 10 times than 1st assignment, this might have happened because of way the data is stored in this dataset, the data 0f 8's and 9's are randomly distributed unlike in the first assignment and due to random distribution the KNN algorithm has random values of 8's and 9's for every prediction instead of a linearly seperables one. One more reason can be large volume of data and with larger volume comes larger distortion (as 8 and 9 are similar) and problem of class overlapping. Also, KFold provides robust model validation and true scoring which wasn't there in assignment1. 

# Question 2


## Dataset Description
Contains players data of Fifa 22 video game : [LINK](https://www.kaggle.com/tolgakurtulus/fifa-22-confirmed-players-dataset)

## Total Samples
15322 Rows

## Total Measurments
17 Columns

## Measurement Description
Contains different attrible of football players like SHOT, PACE, PHYSICAL,REF, KICK, SM, DRIBBLE, PASS, SPEED , AWR, DWR etc. (Total 17)

## Group of Interest (1)
Attacking players (Position) are considered group of interest denoted by 1. Counts : 5426 rows × 17 columns

## Group of Non-Interest (0)
Non-Attacking players (Position) are considered group not of interest denoted by 0. Counts : 9896 rows × 11 columns

## AUC Values
Top - 10 AUC Values
| Feature | AUC |
|:--------|:-----:|
| SHO | 0.879 |
| DEF | 0.201 |
| PAC | 0.778 |
| SM | 0.778 |
| DRI | 0.765 |
| DWR | 0.661 |
| AWR | 0.349 |
| PAS | 0.639 |
| POS | 0.414 |
| KIC | 0.414 |

<br>Complete List of AUC values


| Feature | AUC |
|:------|:-----:|
| SHO | 0.879|
| DEF | 0.201|
| PAC | 0.778 |
| SM  | 0.778 |
| DRI  |0.765 |
| DWR  |0.661 |
| AWR  |0.349|
| PAS  |0.639|
| POS  |0.414|
| KIC  |0.414|
| REF  |0.414|
| HAN  |0.414|
| SPD  |0.414|
| DIV  |0.414|
| PHY  |0.423|
| OVR  |0.535|
| Foot  |0.508|


# Question 3

### Rounded Error Rates Q3
| SVM Linear| SVM RBF | RF | KNN1 | KNN5 | KNN10 |
|:---|:-----:|:-----|:-----:|:--------|:-----:|
| 0.038 | 0.04 | 0.039 | 0.056 | 0.044 | 0.043 |

## A 
No, SVM linear kernel model is the best perfoming model with anerror rate of 0.038 in Q3 (SVM RBF was best performing in Q1), however its only marginally better perfoming than RF and SVM RBF.
<br><br>
Comparing the Error rates of two questions, in Q1 KNN1, 5 and 10 and RF had similar error rates and SVM linear performed the worst and SVM RBF was the best, however in Q3 KNN5 and 10 had similar error rates but KNN1 performed the worst while SVM RBF, SVM Linear and RF has similar error rates.

The dataset does have an impact on the error rates and prediction capabilities of the model. Data distribution and type of data majorly affects the error rate of the model. In SVM linear the classes are seprated by a straight line, In my dataset the linear data present might have helped the model to score better in comparison to first example. Imbalance data (difference between the no of values of Group of interest and group not of interest) is also very large in my dataset which hampers the performance of the models.  Higher standard deviation in dataset also makes the model suffer. For KNNs, higher error rate at lower values of K is mostly due to errorneous/inconsistent testing data and also class overlapping affect their performance. All these points combinely affect a model's result.

