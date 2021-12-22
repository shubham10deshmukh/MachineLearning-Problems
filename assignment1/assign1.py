# ----- PACKAGE IMPORT ---------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


# Method to Run KNN classifier and plot the Error Rate & save the image.
# PARAMETER :
# train_X : X training array, train_Y : Y training array
# test_X : X testing array, test_Y : Y testing array
# file_name : Name of the image file (graph) to be saved.
# RETURN :
# error_rate : A array conatining Error rates.


def model_plotting(train_X, train_Y, test_X, test_Y, file_name) -> None:
    # -------------MODEL-START--------------
    # Citation : www.datascienceplus.com/k-nearest-neighbors-knn-with-python/

    error_rate = []
    for K in range(1, 21):
        knnModel = KNeighborsClassifier(n_neighbors=K).fit(train_X, train_Y)
        Y_predicted = knnModel.predict(test_X)
        error_rate.append(np.mean(Y_predicted != test_Y))

    # -------------PLOTTING-START-----------

    plt.figure(figsize=(10, 8))
    plt.plot(
        range(1, 21),
        error_rate,
        color="black",
        linestyle="dashed",
        marker="X",
        markerfacecolor="red",
        markersize=8,
    )
    plt.title("Error Rate vs. K")
    plt.xlabel("K")
    plt.ylabel("Error Rate")
    plt.savefig(file_name)
    # -------------PLOTTING-END------------

    return error_rate


def question1() -> None:

    inputData = loadmat("NumberRecognition.mat")

    train_8s = inputData["imageArrayTraining8"]
    train_9s = inputData["imageArrayTraining9"]
    test_8s = inputData["imageArrayTesting8"]
    test_9s = inputData["imageArrayTesting9"]

    # Concatenate both the Training data and Testing Data
    # to make a single Training Set, i.e. 8 and 9

    train_X = np.concatenate((train_8s, train_9s), axis=2)
    test_X = np.concatenate((test_8s, test_9s), axis=2)

    # -------------DATA RESHAPING-START--------------

    # Reshape the X Training  data from 3D to 2D inorder to fit it in Model.
    n_samples_train = train_X.shape[2]
    n_features_train = np.prod(train_X.shape[:2])
    train_X = train_X.reshape([n_features_train, n_samples_train])

    # Reshape the X Testing  data from 3D to 2D inorder to fit it in Model.
    n_samples_test = test_X.shape[2]
    n_features_test = np.prod(test_X.shape[:2])
    test_X = test_X.reshape([n_features_test, n_samples_test])

    # -------------LABELING-START--------------

    # Creating lables (TrainY) 8 is represented by 0 and 9 is represented by 1.

    train_8sY = np.zeros(train_8s.shape[-1])
    train_8sY.shape
    train_9sY = np.ones(train_9s.shape[-1])
    train_9sY.shape
    train_Y = np.concatenate((train_8sY, train_9sY), axis=0)

    # Creating lables (TestY) 8 is represented by 0 and 9 is represented by 1.

    test_8s_Y = np.zeros(test_8s.shape[-1])
    test_8s_Y.shape
    test_9s_Y = np.ones(test_9s.shape[-1])
    test_9s_Y.shape
    test_Y = np.concatenate((test_8s_Y, test_9s_Y), axis=0)

    # -------------LABELING-END--------------

    # ----------TRANSPOSING THE DATA----------
    # Transpoing the X data to make match it with the shape of Y data (labels)
    train_X = train_X.transpose(1, 0)  # SHAPE : (1500, 784)
    test_X = test_X.transpose(1, 0)  # SHAPE : , (500, 784)

    error_rate = model_plotting(train_X, train_Y, test_X, test_Y, "knn_q1.png")

    error_rate_percentage = []
    for i in error_rate:
        error_rate_percentage.append(i * 100)

    # --------ERROR.NPY FILE GENERATION------
    save(error_rate_percentage)
    print("Question-1 Successfully Executed")


# ---------------------QUESTION-1-ENDS--------------------


def question2() -> None:
    # --------- PLEASE REFER TO DATASET IN CASE OF QUERY -------------------
    # https://www.kaggle.com/tolgakurtulus/fifa-22-confirmed-players-dataset

    # -----------DATA PREPROCESSING START--------
    data_fifa = pd.read_csv("Fifa22ConfirmedPlayers.csv")

    data_fifa["Foot"] = data_fifa["Foot"].replace(
        to_replace=["Left", "Right"], value=[0, 1]
    )
    data_fifa = data_fifa.drop(columns=["PlayerName"])

    data_fifa["AWR"] = data_fifa["AWR"].replace(np.nan, "NA")
    data_fifa["AWR"] = data_fifa["AWR"].replace(
        to_replace=["High", "Med", "Low", "NA"], value=[0, 1, 2, 3]
    )
    data_fifa["AWR"].isna().sum()

    data_fifa["DWR"] = data_fifa["DWR"].replace(np.nan, "NA")
    data_fifa["DWR"] = data_fifa["DWR"].replace(
        to_replace=["High", "Med", "Low", "NA"], value=[0, 1, 2, 3]
    )
    data_fifa["DWR"].isna().sum()

    data_fifa = data_fifa.replace(np.nan, "NA")
    data_fifa = data_fifa.replace(to_replace=["NA"], value=[0])

    data_fifa["Position"] = data_fifa["Position"].replace(
        ["LF", "RF", "ST", "CF", "RW", "LW", "CAM", "RM", "LM"], 1
    )
    data_fifa["Position"] = data_fifa["Position"].replace(
        ["CB", "RB", "CM", "RWB", "CDM", "LB", "LWB", "GK"], 0
    )

    data_fifa_label = data_fifa["Position"]

    data_fifa = data_fifa.drop(columns=["Position"])

    data_fifa.astype(int)

    # -----------DATA PREPROCESSING ENDS---------------------------

    train_X = data_fifa
    train_Y = data_fifa_label

    auc_features = []

    for columnName in train_X.columns:

        feature = np.array(train_X[columnName])

        auc_features.append(columnName)
        auc_features.append(roc_auc_score(train_Y, feature).round(3))

    auc_features = pd.DataFrame(
        np.array(auc_features).reshape(17, 2), columns=["Features", "AUC"]
    ).sort_values(by=["AUC"], ascending=False)

    print("---- AUC Scores : All the Features/Columns ----")
    print(auc_features)

    print("---- AUC scores : Top 10 Features/Columns ----")
    print(auc_features.head(10))
    print("Question-2 Successfully Executed")


# ---------------------QUESTION-2-ENDS--------------------


def question3():

    # -----------DATA PREPROCESSING START--------
    data_fifa = pd.read_csv("Fifa22ConfirmedPlayers.csv")

    data_fifa["Foot"] = data_fifa["Foot"].replace(
        to_replace=["Left", "Right"], value=[0, 1]
    )
    data_fifa = data_fifa.drop(columns=["PlayerName"])

    data_fifa["AWR"] = data_fifa["AWR"].replace(np.nan, "NA")
    data_fifa["AWR"] = data_fifa["AWR"].replace(
        to_replace=["High", "Med", "Low", "NA"], value=[0, 1, 2, 3]
    )

    data_fifa["DWR"] = data_fifa["DWR"].replace(np.nan, "NA")
    data_fifa["DWR"] = data_fifa["DWR"].replace(
        to_replace=["High", "Med", "Low", "NA"], value=[0, 1, 2, 3]
    )

    data_fifa = data_fifa.replace(np.nan, "NA")
    data_fifa = data_fifa.replace(to_replace=["NA"], value=[0])

    data_fifa["Position"] = data_fifa["Position"].replace(
        ["LF", "RF", "ST", "CF", "RW", "LW", "CAM", "RM", "LM"], 1
    )
    data_fifa["Position"] = data_fifa["Position"].replace(
        ["CB", "RB", "CM", "RWB", "CDM", "LB", "LWB", "GK"], 0
    )

    data_fifa_label = data_fifa["Position"]

    data_fifa = data_fifa.drop(columns=["Position"])

    data_fifa.astype(int)

    # -----------DATA PREPROCESSING ENDS---------------------------

    train_X = data_fifa
    train_Y = data_fifa_label

    X_train, X_test, y_train, y_test = train_test_split(
        train_X, train_Y, test_size=0.35, random_state=0
    )

    model_plotting(X_train, y_train, X_test, y_test, "knn_q3.png")
    print("Question-3 Successfully Executed")


# ---------------------QUESTION-3-ENDS--------------------


# IMPLEMENTATION FROM ASSIGNMENT SUBMISSION PAGE
def save(errors) -> None:
    import numpy as np
    from pathlib import Path

    arr = np.array(errors)
    print()
    if len(arr.shape) > 2 or (len(arr.shape) == 2 and 1 not in arr.shape):
        raise ValueError(
            "Invalid output shape. Output should be an array "
            "that can be unambiguously raveled/squeezed."
        )
    if arr.dtype not in [np.float64, np.float32, np.float16]:
        raise ValueError("Your error rates must be stored as float values.")
    arr = arr.ravel()
    if len(arr) != 20 or (arr[0] >= arr[-1]):
        raise ValueError(
            "There should be 20 error values, with the first value "
            "corresponding to k=1, and the last to k=20."
        )
    if arr[-1] >= 2.0:
        raise ValueError(
            "Final array value too large. You have done something "
            "very wrong (probably relating to standardizing)."
        )
    if arr[-1] < 0.8:
        raise ValueError("You have not converted your error rates to percent.")
    outfile = Path("__file__").resolve().parent / "errors.npy"
    np.save(outfile, arr, allow_pickle=False)
    print("Error rates succesfully saved to {outfile }")


if __name__ == "__main__":

    question1()
    question2()
    question3()
