import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_curve, auc


dataset = "dataset.xlsx"
df = pd.read_excel(dataset)

#columns separation
def load_data(dataset):
    X = df[['person_age', 'person_income', 'person_home_ownership',
            'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt']]
    y = df['loan_status']
    return X, y


#preprocessing function
def preprocessing(X):
    categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    encoder = OneHotEncoder(sparse_output=False, drop='first') 
    #unsuitable columns are transformed into binary (so the number of columns is multiplied by the number of possible results)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]),
                             columns=encoder.get_feature_names_out(categorical_columns))
    #process to the imputation
    columns_to_impute = ['loan_int_rate', 'person_emp_length']
    imputer = KNNImputer(n_neighbors=5)
    X[columns_to_impute] = imputer.fit_transform(X[columns_to_impute])

    X_preprocessed = pd.concat([X_encoded, X.drop(columns=categorical_columns)], axis=1)
    return X_preprocessed

X, y = load_data(dataset)
X = preprocessing(X)
#Scaling function 
X_standardized = StandardScaler().fit_transform(X)

#we will separate the training data and test data
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3)

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['Rejected', 'Approved'], filled=True, fontsize=10)
plt.title("Decision Tree")
plt.show()
