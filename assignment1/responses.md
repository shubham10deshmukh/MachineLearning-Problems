# Question 1
## a
The error rate rises at higher values of K due to increasing number of neighbours and the distant ones also affecting the outcome, which causes higher variance in the data and will tend to bais the KNN towards the dominant class and makes the model more complex.

## b
Error rate at lowest value of K is 1. At K = 1, the model is not reliable enough becuase it will only look for its closest neighbour and will leave the rest of the output data which will affect the performance of the predict output.

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
| Feature | AUC |
|:----------------------|:-----:|
| SHO | 0.879 |
| PAC | 0.778 |
| SM | 0.778 |
| DRI | 0.765 |
| DWR | 0.661 |
| PAS | 0.639 |
| OVR | 0.535 |
| Foot | 0.508 |
| PHY | 0.423 |
| HAN | 0.414 |

# Question 3

The profile of K vs Error Rate is very diffrent from the digit recognition dataset, its infact total opposite of it. The new graph has higher error rate at lower values of K and the error rate decreases as the value of K increases. The dataset does have an impact on the plot. Higher error rate at lower values of K is due to errorneous/inconsistent testing data. Data distribution also affects the error rate with the area of higher data points being sensitive and it makes the KNN problematic. Higher standard deviation in dataset also makes the KNN suffer. All these points combinely affect the KNN's result. 