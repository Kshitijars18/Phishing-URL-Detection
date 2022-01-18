######PHISHING URL DETECTION########
#METHODOLOGY
# URL Data-> Data Preprocessing-> Feature Extraction-> Feature Selection->Applying Ensemble & Non Ensemble Techniques-> Evaluation of Results & Output
# Steps In Detail
# 1) URL DATA :-> URL Data is Taken From User
# 2) Data Preprocessing :-> operation :-> data cleaning, data generalization, data sampling using min-max scalar we are standardizing the dataset
# 3) Feature Extraction & Feature Selection - > ALL the factors are important hence all the feature set are kept
# 4) target set:-> -1 phishing url and 1 legit url
# 5) Ensemble and Non Ensemble :-> Ensemble:->Bagging, Random Forest Classifier, Adaboost Classifier, Gradient Classifier
#Non Ensemble Learning :-> Decision Tree, K-Neighbours, Logistic Regression
# 6) Evaluation : Three main comparision : Accuracy Comparision, Training Time, PRF Comparision



# REQUIRED LIBRARIES
import numpy as np
import pandas as pd
from scipy.io import arff
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime


from flask import Flask, render_template,request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        df1 = pd.read_csv(f.filename)
        for col in df1:
            plt.hist(df1[col], bins=3)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.title(col)
            plt.show()
            # print(df1.corr())

        x = df1.iloc[:, 0:31]
        y = df1.loc[:, "Result"]
        train_x, test_x, train_y, test_y = train_test_split(x, y, shuffle=True, test_size=0.2)

        # Bagging Classifier
        time_st_bc = datetime.now()
        bc = BaggingClassifier(n_estimators=15)
        bc.fit(train_x, train_y)
        time_fn_bc = datetime.now()
        predict_y_bc = bc.predict(test_x)
        accuracy_bc = metrics.accuracy_score(test_y, predict_y_bc)
        precision_bc = metrics.precision_score(test_y, predict_y_bc)
        recall_bc = metrics.recall_score(test_y, predict_y_bc)
        f1_score_bc = metrics.f1_score(test_y, predict_y_bc)
        confusion_matrix_bc = metrics.plot_confusion_matrix(bc, test_x, test_y)
        plt.title("Bagging Classifier")
        plt.show()
        time_bc = time_fn_bc - time_st_bc

        # Random Forest Classifer
        time_st_rfc = datetime.now()
        rfc = RandomForestClassifier(n_estimators=15)
        rfc.fit(train_x, train_y)
        time_fn_rfc = datetime.now()
        predict_y_rfc = rfc.predict(test_x)
        accuracy_rfc = metrics.accuracy_score(test_y, predict_y_rfc)
        precision_rfc = metrics.precision_score(test_y, predict_y_rfc)
        recall_rfc = metrics.recall_score(test_y, predict_y_rfc)
        f1_score_rfc = metrics.f1_score(test_y, predict_y_rfc)
        confusion_matrix_rfc = metrics.plot_confusion_matrix(rfc, test_x, test_y)
        plt.title("Random Forest Classifier")
        plt.show()
        time_rfc = time_fn_rfc - time_st_rfc

        # Adaboost Classifier
        time_st_ac = datetime.now()
        ac = AdaBoostClassifier(n_estimators=15)
        ac.fit(train_x, train_y)
        time_fn_ac = datetime.now()
        predict_y_ac = ac.predict(test_x)
        accuracy_ac = metrics.accuracy_score(test_y, predict_y_ac)
        precision_ac = metrics.precision_score(test_y, predict_y_ac)
        recall_ac = metrics.recall_score(test_y, predict_y_ac)
        f1_score_ac = metrics.f1_score(test_y, predict_y_ac)
        confusion_matrix_ac = metrics.plot_confusion_matrix(ac, test_x, test_y)
        plt.title("Adaboost Classifier")
        plt.show()
        time_ac = time_fn_ac - time_st_ac

        # Gradient Boosting Classifier
        time_st_gbc = datetime.now()
        gbc = GradientBoostingClassifier(n_estimators=15)
        gbc.fit(train_x, train_y)
        time_fn_gbc = datetime.now()
        predict_y_gbc = gbc.predict(test_x)
        accuracy_gbc = metrics.accuracy_score(test_y, predict_y_gbc)
        precision_gbc = metrics.precision_score(test_y, predict_y_gbc)
        recall_gbc = metrics.recall_score(test_y, predict_y_gbc)
        f1_score_gbc = metrics.f1_score(test_y, predict_y_gbc)
        confusion_matrix_gbc = metrics.plot_confusion_matrix(gbc, test_x, test_y)
        plt.title("Gradient Boosting Classifier")
        plt.show()
        time_gbc = time_fn_gbc - time_st_gbc

        # Decision Tree Classifier
        time_st_dt = datetime.now()
        dt = DecisionTreeClassifier(max_depth=2)
        dt.fit(train_x, train_y)
        time_fn_dt = datetime.now()
        predict_y_dt = bc.predict(test_x)
        accuracy_dt = metrics.accuracy_score(test_y, predict_y_dt)
        precision_dt = metrics.precision_score(test_y, predict_y_dt)
        recall_dt = metrics.recall_score(test_y, predict_y_dt)
        f1_score_dt = metrics.f1_score(test_y, predict_y_dt)
        confusion_matrix_dt = metrics.plot_confusion_matrix(dt, test_x, test_y)
        plt.title("Decision Tree Classifier")
        plt.show()
        time_dt = time_fn_dt - time_st_dt

        # KNeighbors Classification
        time_st_kn = datetime.now()
        kn = KNeighborsClassifier(n_neighbors=15)
        kn.fit(train_x, train_y)
        time_fn_kn = datetime.now()
        predict_y_kn = kn.predict(test_x)
        accuracy_kn = metrics.accuracy_score(test_y, predict_y_kn)
        precision_kn = metrics.precision_score(test_y, predict_y_kn)
        recall_kn = metrics.recall_score(test_y, predict_y_kn)
        f1_score_kn = metrics.f1_score(test_y, predict_y_kn)
        confusion_matrix_kn = metrics.plot_confusion_matrix(kn, test_x, test_y)
        plt.title("KNeighbors Classification")
        plt.show()
        time_kn = time_fn_kn - time_st_kn

        # Logistic Regression
        time_st_lr = datetime.now()
        lr = LogisticRegression(max_iter=1000)
        lr.fit(train_x, train_y)
        time_fn_lr = datetime.now()
        predict_y_lr = lr.predict(test_x)
        accuracy_lr = metrics.accuracy_score(test_y, predict_y_lr)
        precision_lr = metrics.precision_score(test_y, predict_y_lr)
        recall_lr = metrics.recall_score(test_y, predict_y_lr)
        f1_score_lr = metrics.f1_score(test_y, predict_y_lr)
        confusion_matrix_lr = metrics.plot_confusion_matrix(lr, test_x, test_y)
        plt.title("Logistic Regression")
        plt.show()
        time_lr = time_fn_lr - time_st_lr

        # Measure for Accuracy
        accuracy = {
            "BC": accuracy_bc,
            "RFC": accuracy_rfc,
            "AD": accuracy_ac,
            "GC": accuracy_gbc,
            "DT": accuracy_dt,
            "KNC": accuracy_kn,
            "LR": accuracy_lr,
        }
        precision = {
            "BC": precision_bc,
            "RFC": precision_rfc,
            "AD": precision_ac,
            "GC": precision_gbc,
            "DT": precision_dt,
            "KNC": precision_kn,
            "LR": precision_lr,

        }
        recall = {
            "BC": recall_bc,
            "RFC": recall_rfc,
            "AD": recall_ac,
            "GBC": recall_ac,
            "DT": recall_dt,
            "KNC": recall_kn,
            "LR": recall_lr,

        }
        f1_score = {
            "BC": f1_score_bc,
            "RFC": f1_score_rfc,
            "AD": f1_score_ac,
            "GBC": f1_score_gbc,
            "DT": f1_score_dt,
            "KN": f1_score_kn,
            "LR": f1_score_lr,
        }
        times = {
            "BC": time_bc.microseconds,
            "RFC": time_rfc.microseconds,
            "AD": time_ac.microseconds,
            "GBC": time_gbc.microseconds,
            "DT": time_dt.microseconds,
            "KN": time_kn.microseconds,
            "LR": time_lr.microseconds,
        }
        print("accuracy")
        print(pd.DataFrame(accuracy.items()))
        print("precision")
        print(pd.DataFrame(precision.items()))
        print("recall")
        print(pd.DataFrame(recall.items()))
        print("f1_score")
        print(pd.DataFrame(f1_score.items()))
        print("times")
        print(pd.DataFrame(times.items()))

        pos = [i for i, _ in enumerate(accuracy)]
        plt.figure(figsize=(10, 7))
        plt.bar(pos, accuracy.values(), color='green')
        plt.xticks(pos, accuracy.keys())
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.show()

        pos = [i for i, _ in enumerate(precision)]
        plt.figure(figsize=(10, 7))
        plt.bar(pos, precision.values(), color='orange')
        plt.xticks(pos, accuracy.keys())
        plt.xlabel("Model")
        plt.ylabel("Precision")
        plt.title("Precision")
        plt.show()

        pos = [i for i, _ in enumerate(recall)]
        plt.figure(figsize=(10, 7))
        plt.bar(pos, recall.values(), color='yellow')
        plt.xticks(pos, recall.keys())
        plt.xlabel("Model")
        plt.ylabel("Recall")
        plt.title("Recall")
        plt.show()

        pos = [i for i, _ in enumerate(f1_score)]
        plt.figure(figsize=(10, 7))
        plt.bar(pos, f1_score.values(), color='red')
        plt.xticks(pos, f1_score.keys())
        plt.xlabel("Model")
        plt.ylabel("F1 Score")
        plt.title("F1 Score")
        plt.show()

        pos = [i for i, _ in enumerate(times)]
        plt.figure(figsize=(10, 7))
        plt.bar(pos, times.values(), color='blue')
        plt.xticks(pos, times.keys())
        plt.xlabel("Model")
        plt.ylabel("Microsecond")
        plt.title("Time Taken")
        plt.show()

    return render_template("success.html", name=f.filename)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
