import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
    columns_to_impute = ['loan_int_rate', 'person_emp_length']
    imputer = KNNImputer(n_neighbors=5)
    X[columns_to_impute] = imputer.fit_transform(X[columns_to_impute])

    X_preprocessed = pd.concat([X_encoded, X.drop(columns=categorical_columns)], axis=1)
    return X_preprocessed

X, y = load_data(dataset)
X = preprocessing(X)
X_standardized = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3)

rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], feature_names=X.columns, class_names=['Rejected', 'Approved'], filled=True, fontsize=10)
plt.title("Example of an Random Forest")
plt.show()

#Metrics
# Precision, Recall, F1-score
print("",classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# Calcul ROC
fpr, tpr, thresholds = roc_curve(y_test, dt.predict_proba(X_test)[:, 1])
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

#confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Random Forest)")
plt.show()