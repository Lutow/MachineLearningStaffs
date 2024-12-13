import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

dataset = "dataset.xlsx"
df = pd.read_excel(dataset)

def load_data(dataset):
    X = df[['person_age', 'person_income', 'person_home_ownership','person_emp_length',
            'loan_intent', 'loan_grade', 'loan_amnt','loan_int_rate', 'loan_percent_income',
            'cb_person_default_on_file','cb_person_cred_hist_length']]
    y = df['loan_status']
    return X, y

def preprocessing(X):
    categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    encoder = OneHotEncoder(sparse_output=False, drop='first') 
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]),
                             columns=encoder.get_feature_names_out(categorical_columns))
    #imputation
    columns_to_impute = ['loan_int_rate', 'person_emp_length']
    imputer = KNNImputer(n_neighbors=5)
    X[columns_to_impute] = imputer.fit_transform(X[columns_to_impute])

    X_preprocessed = pd.concat([X_encoded, X.drop(columns=categorical_columns)], axis=1)
    return X_preprocessed

X, y = load_data(dataset)
X_process = preprocessing(X)
X_standardized = StandardScaler().fit_transform(X_process)

#we will reduce the dimension to represent it so will use the analyze in principal components 
#This simplifies the data, eliminates correlations and reduces the risk of overfitting, it's also reducing data to 2 dimensions like that we can plot it on a 2D graph to observe it
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3)

k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

xx, yy = np.meshgrid(np.arange(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 0.1),
                     np.arange(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolor='k' )
plt.title(f"KNN Decision Boundaries (k={k})")
plt.show()

print("",classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
# Calcul ROC
fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
#courbe ROC
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="No Skill (Random Guess)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()
#matrix confusion
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (KNN)")
plt.show()
