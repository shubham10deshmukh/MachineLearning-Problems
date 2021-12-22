from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def question1() -> None:
    input_data = loadmat("NumberRecognitionBigger.mat")

    X = input_data["X"]
    y = input_data["y"]

    # -- CONVERTING shape of Image data from 3D to 2D array--
    n_samples_train = X.shape[2]
    n_features_train = np.prod(X.shape[:2])
    X = X.reshape([n_features_train, n_samples_train])

    # -- TRANSPOSING & Making them DataFrame --
    X = pd.DataFrame(X.transpose())
    y = pd.DataFrame(y.transpose(), columns=["label"])

    # -- CONCATINATING X and Y --
    data_Xy = pd.concat([X, y], axis=1,)

    # -- DROPPING rows containing (0-7)--
    for i in range(0, 8):
        data_Xy = data_Xy.drop(data_Xy[data_Xy.label == i].index)

    train_y = data_Xy["label"]
    train_X = data_Xy.drop(columns=["label"])

    # -- CALLING Modelling and Scoring Method--
    kfold_scores = model_scoring(train_X, train_y)

    # -- SAVING JSON file --
    save_mnist_kfold(kfold_scores)

    print("Question-1 Successfully Executed")


def question2() -> None:

    # ----CALLING THE DATA IMPORT & PREPROCESSING METHOD----
    data = load()

    train_X = data["x"]
    train_y = data["y"]

    auc_features = []
    column_names = []

    for columnName in train_X.columns:

        feature = np.array(train_X[columnName])
        auc_features.append(roc_auc_score(train_y, feature))
        column_names.append(columnName)

    sorted_auc = sort_auc(auc_features, column_names)

    # -- SAVING JSON file --
    sorted_auc.to_json(Path("__file__").resolve().parent / "aucs.json")

    print("---- Rounded AUC Scores : All the Features/Columns ----")
    print(sorted_auc.round(3))

    print("---- Rounded AUC scores : Top 10 Features/Columns ----")
    print(sorted_auc.round(3).head(10))

    print("Question-2 Successfully Executed")


def question3() -> None:

    # ----CALLING THE DATA IMPORT & PREPROCESSING METHOD----
    data = load()

    train_X = data["x"]
    train_y = data["y"]

    # -- CALLING THE KFOLD AND MODELLING METHOD--
    kfold_scores = model_scoring(train_X, train_y)

    # -- SAVING JSON file --
    save_data_kfold(kfold_scores)

    print("Question-3 Successfully Executed")


# ------------KFOLD, MODELLING AND SCORING METHOD------------
# PARAMETERS: X data and Y data  ----------------------------
# RETURNS   : kfold_scores DataFrame ------------------------


def model_scoring(train_X, train_y) -> None:

    # -- Array to Store error rates of different methods--
    error_rate_SVM_linear = []
    error_rate_SVM_RBF = []
    error_rate_RF = []
    error_rate_KNN1 = []
    error_rate_KNN5 = []
    error_rate_KNN10 = []

    # -- KFold (N = 5)
    SKF = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    SKF.get_n_splits(train_X, train_y)

    # -- Initializing all the Models--
    svm_linear = SVC(gamma="scale", kernel="linear", random_state=42)
    svm_rbf = SVC(gamma="scale", kernel="rbf", random_state=42)
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn10 = KNeighborsClassifier(n_neighbors=10)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    for train_index, test_index in SKF.split(train_X, train_y):

        X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

        svm_linear.fit(X_train, y_train)
        error_rate_SVM_linear.append(1 - svm_linear.score(X_test, y_test))

        svm_rbf.fit(X_train, y_train)
        error_rate_SVM_RBF.append(1 - svm_rbf.score(X_test, y_test))

        rf.fit(X_train, y_train)
        error_rate_RF.append(1 - rf.score(X_test, y_test))

        knn1.fit(X_train, y_train)
        test1 = knn1.score(X_test, y_test)
        error_rate_KNN1.append(1 - knn1.score(X_test, y_test))

        knn5.fit(X_train, y_train)
        error_rate_KNN5.append(1 - knn5.score(X_test, y_test))

        knn10.fit(X_train, y_train)
        error_rate_KNN10.append(1 - knn10.score(X_test, y_test))

    scores = [
        np.mean(error_rate_SVM_linear).round(3),
        np.mean(error_rate_SVM_RBF).round(3),
        np.mean(error_rate_RF).round(3),
        np.mean(error_rate_KNN1).round(3),
        np.mean(error_rate_KNN5).round(3),
        np.mean(error_rate_KNN10).round(3),
    ]

    # -- Error Rate DataFrame--
    kfold_scores = pd.DataFrame(
        data=[scores],
        columns=["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"],
        index=["err"],
    )

    return kfold_scores


# -------------------------SORT AUC METHOD-----------------------------
# PARAMETERS
# auc_scores : Conatains list of auc scores                  ----------
# column_names : Conatains list of features name(Column name) ---------
# RETURNS : A Dataframe containing sorted(further from 0.5) AUC values.


def sort_auc(auc_scores, column_names) -> None:
    auc_array = []

    for vals in auc_scores:
        if vals < 0.5:
            vals = 1 - vals

        auc_array.append(vals)

    # A DF with a column AUC_adjusted which contains adjusted AUC values.
    # Eg. AUC score of 0.3 is stated as 1 - 0.3 = 0.7 (Furtherest from 0.5)
    # The column is present only for sorting and doesnt change the actual AUC.

    auc_adjustedDF = (
        pd.DataFrame(
            {"Features": column_names, "AUC": auc_scores, "AUC_new": auc_array}
        )
        .sort_values(by=["AUC_new"], ascending=False)
        .drop(columns="AUC_new")
    )

    return auc_adjustedDF


# ----------IMPORITNG & PREPROCESSING DATA's METHOD----------
# RETURNS   : Processed DataFrame (X and Y combined) --------


def load() -> None:

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

    data = {"x": data_fifa, "y": data_fifa_label}

    # -----------DATA PREPROCESSING ENDS---------------------------

    return data


# ------------------- METHODS FOR EXTERNAL FILES--------------------


def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:
    from pathlib import Path

    import numpy as np
    from pandas import DataFrame

    COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError(
            "Argument `kfold_scores` to `save` must be a pandas DataFrame."
        )
    if kfold_scores.shape != (1, 6):
        raise ValueError("DataFrame must have 1 row and 6 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError(
            "Row has bad index name. Use `kfold_score.index = ['err']` to fix."
        )

    if np.min(df.values) < 0 or np.max(df.values) > 0.10:
        raise ValueError(
            "Your K-Fold error rates are too extreme."
            " Ensure they are the raw error rates,\r\n"
            "and NOT percentage error rates."
            " Also ensure your DataFrame contains error rates,\r\n"
            "and not accuracies. If you are sure you have"
            " not made either of the above mistakes,\r\n"
            "there is probably something else wrong with"
            " your code. Contact the TA for help.\r\n"
        )

    if df.loc["err", "svm_linear"] > 0.07:
        raise ValueError(
            "Your svm_linear error rate is too high."
            " There is likely an error in your code."
        )
    if df.loc["err", "svm_rbf"] > 0.03:
        raise ValueError(
            "Your svm_rbf error rate is too high."
            " There is likely an error in your code."
        )
    if df.loc["err", "rf"] > 0.05:
        raise ValueError(
            "Your Random Forest error rate is too high."
            " There is likely an error in your code."
        )
    if df.loc["err", ["knn1", "knn5", "knn10"]].min() > 0.04:
        raise ValueError(
            "One of your KNN error rates is too high."
            " There is likely an error in your code."
        )

    outfile = Path("__file__").resolve().parent / "kfold_mnist.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")


def save_data_kfold(kfold_scores: pd.DataFrame) -> None:
    from pathlib import Path

    import numpy as np
    from pandas import DataFrame

    COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError(
            "Argument `kfold_scores` to `save` must be a pandas DataFrame."
        )
    if kfold_scores.shape != (1, 6):
        raise ValueError("DataFrame must have 1 row and 6 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError(
            "Row has bad index name. Use `kfold_score.index = ['err']` to fix."
        )

    outfile = Path("__file__").resolve().parent / "kfold_data.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for individual dataset saved to {outfile}")


if __name__ == "__main__":

    question1()
    question2()
    question3()
