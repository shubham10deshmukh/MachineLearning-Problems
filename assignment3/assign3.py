import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    ReLU,
)


# Class Containing Static Methods for all the Machine Learning Models used
class My_Models:

    # ------------CNN Model METHOD  -------------------------------------
    # PARAMETERS:
    # X_train, X_test, y_train, y_test, error_rates( initialized list) --
    # RETURNS   : error_rates(list)  ------------------------------------
    # Returns error rates of CNN ----------------------------------------

    # REFERENCE : Assignment PDF & Moodle Tutorial
    @staticmethod
    def CNN_model(X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates) -> None:

        NUM_CLASSES = 10
        INPUT_SHAPE = (28, 28, 1)
        EPOCHS = 15

        X_train_KF = X_train_KF.astype("float32")
        X_test_KF = X_test_KF.astype("float32")
        X_train_KF /= 255
        X_test_KF /= 255

        X_train_KF = np.expand_dims(X_train_KF, -1)
        X_test_KF = np.expand_dims(X_test_KF, -1)

        CNN = Sequential()
        CNN.add(BatchNormalization())
        CNN.add(
            Conv2D(
                5,
                kernel_size=3,
                input_shape=INPUT_SHAPE,
                activation="linear",
                data_format="channels_last",
            )
        )
        CNN.add(ReLU())
        CNN.add(Flatten())
        CNN.add(Dense(units=10, activation="softmax"))
        CNN.compile(
            optimizer=keras.optimizers.SGD(momentum=0.9, learning_rate=0.001),
            loss=keras.losses.mean_squared_error,
            metrics=["accuracy"],
        )

        y_train_KF = keras.utils.to_categorical(y_train_KF, NUM_CLASSES)
        y_test_KF = keras.utils.to_categorical(y_test_KF, NUM_CLASSES)

        CNN.fit(X_train_KF, y_train_KF, epochs=EPOCHS, verbose=1)

        CNN_score = CNN.evaluate(X_test_KF, y_test_KF, verbose=1)

        error_rates["CNN"].append(1 - CNN_score[1])

        return error_rates

    # ------------KNN Models METHOD  ------------------------------------
    # PARAMETERS:
    # X_train, X_test, y_train, y_test, error_rates( initialized list) --
    # RETURNS   : error_rates(list)  ------------------------------------
    # Returns error rates of KNN-1, KNN-5, KNN-10 -----------------------

    @staticmethod
    def KNN_models(X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates) -> None:

        # -- Initializing all the KNNs--
        knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train_KF, y_train_KF)
        error_rates["KNN1"].append(
            1 - accuracy_score(knn1.predict(X_test_KF), y_test_KF)
        )

        knn5 = KNeighborsClassifier(n_neighbors=5).fit(X_train_KF, y_train_KF)
        error_rates["KNN5"].append(
            1 - accuracy_score(knn5.predict(X_test_KF), y_test_KF)
        )

        knn10 = KNeighborsClassifier(n_neighbors=10).fit(X_train_KF, y_train_KF)
        error_rates["KNN10"].append(
            1 - accuracy_score(knn10.predict(X_test_KF), y_test_KF)
        )

        return error_rates

    # ------------CNN Model (MY BESY SCORE) METHOD  -------------------
    # PARAMETERS:
    # X_train, X_test,
    # y_train, y_test(or None), error_rates( initialized list/ None) --
    # RETURNS   : error_rates(list)  ----------------------------------
    # Returns error rates of CNN/ None in Case of Predicting Lables ---
    # Sending None for parameter y_test and error_rates
    # when predicting lables for X_test --------------------------------
    # REFERENCE: github.com/mohamed0998/CNN-MNIST-dataset-/blob/master/mnist.ipynb

    @staticmethod
    def CNN_model_best(
        X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates
    ) -> None:

        NUM_CLASSES = 10
        INPUT_SHAPE = (28, 28, 1)
        OUT_CHANNELS = 50
        EPOCHS = 15

        CNN = Sequential()

        # -- Converting to float and in the range of 0-1. --
        X_train_KF = X_train_KF.astype("float32")
        X_test_KF = X_test_KF.astype("float32")
        X_train_KF /= 255
        X_test_KF /= 255

        # -- Expanding to 4d from 3d --.
        X_train_KF = np.expand_dims(X_train_KF, -1)
        X_test_KF = np.expand_dims(X_test_KF, -1)

        CNN.add(BatchNormalization())
        CNN.add(Conv2D(OUT_CHANNELS, 5, activation="relu", input_shape=INPUT_SHAPE)),
        CNN.add(MaxPooling2D(pool_size=(2, 2)))
        CNN.add(Conv2D(OUT_CHANNELS, 3, activation="relu"))
        CNN.add(MaxPooling2D(pool_size=(2, 2)))
        CNN.add(Flatten())
        CNN.add(Dense(NUM_CLASSES))

        CNN.compile(
            optimizer="adam", loss=keras.losses.mean_squared_error, metrics=["accuracy"]
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="accuracy",
            mode="auto",
            min_delta=0,
            patience=2,
            verbose=0,
            restore_best_weights=True,
        )

        y_train_KF = keras.utils.to_categorical(y_train_KF, NUM_CLASSES)

        # -- Condition to check if method is called for evaluating error rate.
        # -- Chaning y_test to categorical value (reshaping according to class)
        if y_test_KF is not None:
            y_test_KF = keras.utils.to_categorical(y_test_KF, NUM_CLASSES)

        CNN.fit(
            X_train_KF, y_train_KF, epochs=EPOCHS, verbose=2, callbacks=[early_stopping]
        )

        # -- Condition to check if method is called for evaluating error rate.
        if y_test_KF is not None:
            CNN_score = CNN.evaluate(X_test_KF, y_test_KF, verbose=1)
            error_rates["CNN_best"].append(1 - CNN_score[1])

        # -- Condition to check if the method is called for predicting Xtest.
        if y_test_KF is None:
            y_predicted = CNN.predict(X_test_KF)
            # y_predicted *= 255
            y_predicted = np.uint8(np.argmax(y_predicted * 255, axis=-1))
            print("Prediction Successful")
            np.save(
                Path(__file__).resolve().parent / "predictions.npy",
                y_predicted,
                allow_pickle=False,
                fix_imports=False,
            )

        return error_rates

    # ------------ANN Models METHOD  --------------------------------------
    # PARAMETERS:
    # X_train, X_test, y_train, y_test, error_rates_ANNs(initialized list) -
    # RETURNS   : error_rates(list)  --------------------------------------
    # Returns error rates of ANN1, ANN2, ANN3, ANN4 -----------------------

    @staticmethod
    def ANN_model(
        X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates_ANNs
    ) -> None:

        ANN = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=5,
            activation="tanh",
            solver="lbfgs",
            max_iter=2000,
        ).fit(X_train_KF, y_train_KF)

        error_rates_ANNs["ANN1"].append(
            1 - accuracy_score(ANN.predict(X_test_KF), y_test_KF)
        )

        ANN = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=5,
            activation="relu",
            solver="sgd",
            learning_rate="adaptive",
            max_iter=2000,
        ).fit(X_train_KF, y_train_KF)

        error_rates_ANNs["ANN2"].append(
            1 - accuracy_score(ANN.predict(X_test_KF), y_test_KF)
        )

        ANN = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=(100,),
            alpha=0.05,
            activation="tanh",
            solver="adam",
            max_iter=2000,
        ).fit(X_train_KF, y_train_KF)

        error_rates_ANNs["ANN3"].append(
            1 - accuracy_score(ANN.predict(X_test_KF), y_test_KF)
        )

        ANN = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=38,
            activation="tanh",
            solver="adam",
            max_iter=2000,
        ).fit(X_train_KF, y_train_KF)

        error_rates_ANNs["ANN4"].append(
            1 - accuracy_score(ANN.predict(X_test_KF), y_test_KF)
        )

        return error_rates_ANNs


# Class Containing Static Methods for all the custom ---
# Utlity Methods used in the assignment-----------------


class My_Utilities:

    # ------------RESHAPE DATA FOR CNN METHOD--------------------
    # PARAMETERS: X_train and Xtest  ----------------------------
    # RETURNS   : X_train and Xtest  ----------------------------
    # Reshaped 2d image data back to 3d array for CNN -----------

    # REFERENCE : DEREK's Reshaping tutorial
    @staticmethod
    def rehsape_cnn(X_train_KF, X_test_KF) -> None:

        n_features = int(math.sqrt(X_train_KF.shape[1]))

        n_samples_train = X_train_KF.shape[0]
        X_train_KF = X_train_KF.reshape([n_samples_train, n_features, n_features])

        n_samples_test = X_test_KF.shape[0]
        X_test_KF = X_test_KF.reshape([n_samples_test, n_features, n_features])

        return X_train_KF, X_test_KF

    # ------------DATAFRANE GENERATOR METHOD --------------------
    # PARAMETERS: err_rounded, columns---------------------------
    # RETURNS   : Dataframe is req. format  ---------------------

    @staticmethod
    def df_generator(err_rounded, columns) -> None:
        return pd.DataFrame(data=[err_rounded], columns=columns, index=["err"])

    # -------------------------SORT AUC METHOD-----------------------------
    # PARAMETERS
    # auc_scores : Conatains list of auc scores                  ----------
    # column_names : Conatains list of features name(Column name) ---------
    # RETURNS : A Dataframe containing sorted(further from 0.5) AUC values.

    @staticmethod
    def sort_auc_scores(auc_scores, column_names) -> None:
        temp = []

        for value in auc_scores:
            if value < 0.5:
                value = 1 - value
            temp.append(value)

        # A DF with a column AUC_adjusted which contains adjusted AUC values.
        # Eg. AUC score of 0.3 is stated as 1 - 0.3 = 0.7 (Furtherest from 0.5)
        # The column is present only for sorting and doesnt change the AUC.

        return (
            pd.DataFrame({"Features": column_names, "AUC": auc_scores, "AUC_new": temp})
            .sort_values(by=["AUC_new"], ascending=False)
            .drop(columns="AUC_new")
        )

    # ---------------DATA LOADER AND PREPROCESS METHOD --------------------
    # PARAMETERS : NONE
    # RETURNS: DataFrame containing preprocessed data from the CSV file ---

    @staticmethod
    def load_data() -> None:

        # -----------DATA PREPROCESSING START--------
        data_fifa = pd.read_csv("mydata.csv")

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

        data_fifa.astype(int)

        data_fifa_label = data_fifa["Position"]

        data_fifa = data_fifa.drop(columns=["Position"])

        # -----------DATA PREPROCESSING ENDS---------------------------

        return data_fifa, data_fifa_label


# REFERENCE : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
# REFERENCE : DEREK's Reshaping tutorial
def question1() -> None:
    train_test_data = loadmat("NumberRecognitionBiggest.mat")
    X_train, y_train = train_test_data["X_train"], train_test_data["y_train"]

    # -- CONVERTING shape of Image data from 3D to 2D array--
    X_train = X_train.reshape([X_train.shape[0], np.prod(X_train.shape[1:])])

    y_train = y_train.reshape([y_train.shape[1]])

    # -- Array to Store error rates of different methods--
    error_rates_q1 = {"CNN": [], "KNN1": [], "KNN5": [], "KNN10": []}

    # -- KFold (N = 5)
    SKF = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    SKF.get_n_splits(X_train, y_train)

    for train_index, test_index in SKF.split(X_train, y_train):

        X_train_KF, y_train_KF, X_test_KF, y_test_KF = (
            X_train[train_index],
            y_train[train_index],
            X_train[test_index],
            y_train[test_index],
        )

        error_rates_q1 = My_Models.KNN_models(
            X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates_q1
        )

        X_train_KF, X_test_KF = My_Utilities.rehsape_cnn(X_train_KF, X_test_KF)

        error_rates_q1 = My_Models.CNN_model(
            X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates_q1
        )

    err_rounded = np.round(
        [
            np.mean(error_rates_q1["CNN"]),
            np.mean(error_rates_q1["KNN1"]),
            np.mean(error_rates_q1["KNN5"]),
            np.mean(error_rates_q1["KNN10"]),
        ],
        3,
    )

    q1_kfold_err = My_Utilities.df_generator(
        err_rounded, ["cnn", "knn1", "knn5", "knn10"]
    )
    save_mnist_kfold(q1_kfold_err)

    print(q1_kfold_err)

    print("Question-1 Successfully Executed")


# Same as Assignment 2 & 1 as the dataset is the same
def question2() -> None:

    # ----CALLING THE DATA IMPORT & PREPROCESSING METHOD----
    X_train, y_train = My_Utilities.load_data()

    # = data["x"]
    auc_features = []

    for columnName in X_train.columns:

        auc_features.append(roc_auc_score(y_train, X_train[columnName]))

    sorted_auc = My_Utilities.sort_auc_scores(auc_features, X_train.columns)
    sorted_auc.to_json(Path(__file__).resolve().parent / "aucs.json")

    print("---- Rounded AUC Scores : All the Features/Columns ----")
    print(sorted_auc.round(3))

    print("---- Rounded AUC scores : Top 10 Features/Columns ----")
    print(sorted_auc.round(3).head(10))

    print("Question-2 Successfully Executed")


# REFERENCE : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
def question3() -> None:

    # ----CALLING THE DATA IMPORT & PREPROCESSING METHOD----
    X_train, y_train = My_Utilities.load_data()

    # -- Array to Store error rates of different methods--
    error_rates_ANNs = {"ANN1": [], "ANN2": [], "ANN3": [], "ANN4": []}
    error_rates_KNNs = {"KNN1": [], "KNN5": [], "KNN10": []}

    SKF = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    SKF.get_n_splits(X_train, y_train)

    for train_index, test_index in SKF.split(X_train, y_train):

        X_train_KF, y_train_KF, X_test_KF, y_test_KF = (
            X_train.iloc[train_index],
            y_train.iloc[train_index],
            X_train.iloc[test_index],
            y_train.iloc[test_index],
        )

        error_rates_ANNs = My_Models.ANN_model(
            X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates_ANNs
        )
        error_rates_KNNs = My_Models.KNN_models(
            X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates_KNNs
        )

    err_rounded = np.round(
        [
            np.mean(error_rates_ANNs["ANN1"]),
            np.mean(error_rates_ANNs["ANN2"]),
            np.mean(error_rates_ANNs["ANN3"]),
            np.mean(error_rates_ANNs["ANN4"]),
            np.mean(error_rates_KNNs["KNN1"]),
            np.mean(error_rates_KNNs["KNN5"]),
            np.mean(error_rates_KNNs["KNN10"]),
        ],
        3,
    )

    q3_kfold_err = My_Utilities.df_generator(
        err_rounded, ["ann1", "ann2", "ann3", "ann4", "knn1", "knn5", "knn10"]
    )

    print(q3_kfold_err)

    save_data_kfold(q3_kfold_err)

    print("Question-3 Successfully Executed")


def question4() -> None:

    train_test_data = loadmat("NumberRecognitionBiggest.mat")
    X_train, y_train, X_test = (
        train_test_data["X_train"],
        train_test_data["y_train"],
        train_test_data["X_test"],
    )

    n_samples_train = X_train.shape[0]
    n_features_train = np.prod(X_train.shape[1:])
    X_train = X_train.reshape(
        [
            n_samples_train,
            n_features_train,
        ]
    )

    y_train = y_train.reshape([y_train.shape[1]])

    # -- Array to Store error rates of different methods--
    error_rates_q4 = {"CNN_best": [], "CNN": [], "KNN1": [], "KNN5": [], "KNN10": []}

    # -- KFold (N = 5)
    SKF = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    SKF.get_n_splits(X_train, y_train)

    for train_index, test_index in SKF.split(X_train, y_train):

        X_train_KF, y_train_KF, X_test_KF, y_test_KF = (
            X_train[train_index],
            y_train[train_index],
            X_train[test_index],
            y_train[test_index],
        )

        error_rates_q4 = My_Models.KNN_models(
            X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates_q4
        )

        X_train_KF, X_test_KF = My_Utilities.rehsape_cnn(X_train_KF, X_test_KF)

        error_rates_q4 = My_Models.CNN_model(
            X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates_q4
        )

        error_rates_q4 = My_Models.CNN_model_best(
            X_train_KF, y_train_KF, X_test_KF, y_test_KF, error_rates_q4
        )

    err_rounded = np.round(
        [
            np.mean(error_rates_q4["CNN_best"]),
            np.mean(error_rates_q4["CNN"]),
            np.mean(error_rates_q4["KNN1"]),
            np.mean(error_rates_q4["KNN5"]),
            np.mean(error_rates_q4["KNN10"]),
        ],
        3,
    )

    q4_kfold_err = My_Utilities.df_generator(
        err_rounded, ["cnn_best", "cnn", "knn1", "knn5", "knn10"]
    )

    np.save(
        Path(__file__).resolve().parent / "kfold_cnn.npy",
        float(np.mean(error_rates_q4["CNN_best"]).round(4)),
        allow_pickle=False,
        fix_imports=False,
    )
    print(q4_kfold_err)

    # -- Reshaping Data from 2d to 3d for CNN prediction --
    # X_train, X_test = My_Utilities.rehsape_cnn(X_train, X_test)

    # -- Predicting for X_test, method return NONE when predicting --
    # My_Models.CNN_model_best(X_train, y_train, X_test, y_test_KF=None, error_rates=None)

    print("Question-4 Successfully Executed")


# REFERENCE: MOODLE SUBMISSION PAGE
def save_data_kfold(kfold_scores: pd.DataFrame) -> None:
    from pathlib import Path

    import numpy as np
    from pandas import DataFrame

    KNN_COLS = sorted(["knn1", "knn5", "knn10"])
    df = kfold_scores
    for knn_col in KNN_COLS:
        if knn_col not in df.columns:
            raise ValueError(
                "Your DataFrame is missing a KNN error rate or is misnamed."
            )
        if not isinstance(df, DataFrame):
            raise ValueError(
                "Argument `kfold_scores` to `save` must be a pandas DataFrame."
            )
        if not df.index.values[0] == "err":
            raise ValueError(
                "Row has bad index name. Use `kfold_score.index = ['err']` to fix."
            )

    outfile = Path(__file__).resolve().parent / "kfold_data.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for individual dataset successfully saved to {outfile}")


# REFERENCE: MOODLE SUBMISSION PAGE
def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:
    from pathlib import Path

    import numpy as np
    from pandas import DataFrame

    COLS = sorted(["cnn", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError(
            "Argument `kfold_scores` to `save` must be a pandas DataFrame."
        )
    if kfold_scores.shape != (1, 4):
        raise ValueError("DataFrame must have 1 row and 4 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError(
            "Row has bad index name. Use `kfold_scores.index = ['err']` to fix."
        )

    if df.loc["err", ["knn1", "knn5", "knn10"]].min() > 0.06:
        raise ValueError(
            "One of your KNN error rates is likely too high. There is likely an error in your code."
        )

    outfile = Path(__file__).resolve().parent / "kfold_mnist.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")


if __name__ == "__main__":

    # question1()
    question2()
    # question3()
    # question4()
