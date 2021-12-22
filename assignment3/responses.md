
# Question 1

### Rounded Error Rates Q1
| CNN | KNN1 | KNN5 | KNN10 |
|:----|:-----:|:-----|:-----:|
| 0.083 | 0.034 | 0.035 | 0.039 |

## A

KNN with K=1 performed the best in qestion 1 with error rate = 0.034, however KNN5 and KNN10 were pretty close to KNN1 with CNN being the worst performing one with error rate of 0.083.

## B
CNN is a class of deep neural network mostly applied for analyzing images. Its built around layers, between the input and output layers there can be unlimited number of hidden layers with their respective activation function which are interconnected and passes the value sequentially from one layer to another and eventually to the output layers. The performace of any CNN depends on the number of relevant layers, its complexity and its configurations, in our case we used very few number of layers for the size of the dataset, adding more relavent layers and giving appropriate configuration to the CNN will help in increasing the performance and score, Layer like Maxpooling can help increase the accuracy but we haven't used it in question1, A non linear activation also improves a model's performance, also we can use same layers multiple times to refinely train the model as done in the question-4,  also KNN5 and KNN10 have higher error rater but the margin is insignificant and this might have happend because of KNN-1 considers only one nearest neighbour leading to overfitting (non-reliable) and giving better scores, while with increasing K values the model considers more samples and noise can get increased leading to higher error rate, to add up the error rates aren't significantly distant in KNNs and the overall the dataset has multiclass output and is large in size which always makes it difficult for the model to predict and hampers the error rates also, kfold validation is strict validation mechanism which makes it the model robust and harder for the model to score leading to overall lesser score in every model.  
 

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
| ANN1 | ANN2 | ANN3 | ANN4 | KNN1 | KNN5 | KNN10 |
|:---|:-----:|:-----|:-----:|:--------|:-----:|:-----:|
| 0.354 | 0.041 | 0.039 |0.038 | 0.056 | 0.044 | 0.043 |


ANN4 worked the best in question 3 for my dataset with error rate of 0.038 however ANN3 and ANN2 where only a tad higher than ANN4, below is my ANN architecture (MLPClassifier).
<br>
|  hidden_layer_sizes | activation | solver  | max_iter | Random State|
|:---|:-----:|:-----|:-----:|:--------|
 | 38 | "tanh" | "adam" | 2000 | 42|


<br><br>
Any model has impact due to data, one of reasons of change in error rates is because of the dataset and its consistency, I tried 4 versions of ANN using MLP classifier from sklearn, All the changes I did was by stuyding sklearn documentation. 
I varied the aplha value which is to tackle overfitting, the default value of 0.0001 worked the best for me because of the my small size of dataset which might have not required overfitting penalization. Increasing Alpha for tackling overfitting icreased by error rates and also increasing the hidden layers hampered my score, I tried changing the hidden layers which is the number of neurons in the hidden layer but the hidden layer = 38 worked best for me. The number of hidden layers depend on the no of input size (features) + output size, Small or large values of hidden layers causes training and generalization error due to both bias and underfitting. A good activation funtion will allow model to train better and efficiently, using a non linear function always better, ReLU and Tanh are most often used, Tanh are more expensive and is from (-1,1) it out performed ReLU in my case. lbfgs solver is fast but used for smaller datasets and it worked bad in my case, so I tried SGD and adam which workes with larger samples, Adam is an extention to SGD and I got similar results for both of them, however my best ANN was with adam. Also, I have used random_state to get a reproducibile code. All these factors contribute towards a model's performance along with the datasets, its volume, its type and consistency etc. 

# Question 4

### Rounded Error Rate Comparision for Q4

|  cnn_best  |   cnn  |  knn1 |   knn5 |  knn10| 
|:-----|:-----:|:--------|:-----:|:-----:|
| 0.012 | 0.082 | 0.034|  0.035 | 0.039|

<br><br>
My best architechture for CNN (cnn_best) got an error rate of 0.012 which is significant improvement over the basic CNN and also much higher than all the three KNNs we have used in the question. Below is the architecture for my CNN:
<br><br>
Its a sequential CNN.<br><br>
BatchNormalization <br>
Conv2D : Out Channels = 50, Kernal = 5, activation="relu", input_shape=28,28,1 <br>
MaxPooling2D : pool_size=(2, 2) <br>
Conv2D : Out Channels = 50, Kernal = 3, activation="relu" <br>
MaxPooling2D: pool_size=(2, 2) <br>
Flattening <br>
Dense : Num Classes = 10 <br>
optimizer : Adam <br>
loss : mean_squared_error<br>
metrics : Accuracy<br>


