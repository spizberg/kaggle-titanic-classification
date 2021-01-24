import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("train.csv")

# Split dataset in trainset and validset
y = train_data.Survived
X = train_data.drop(["Survived"],axis=1)


# Preprocessing with custom transformer

class CustomTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        super(CustomTransformer,self).__init__()
        self.embarked_values = ["S","C","Q","Unknown"]
        self.numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare','Fare_Per_Person']
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        dataset = X.copy()
        dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch']
        dataset['Fare_Per_Person']=dataset['Fare']/(dataset['Family_Size']+1)
        dataset["Male"] = dataset["Sex"].map(lambda x: 1 if x=="male" else 0)
        dataset["Female"] = dataset["Sex"].map(lambda x: 1 if x=="female" else 0)
        dataset["Cabin"] = dataset["Cabin"].map(lambda x: 0 if x is np.nan else 1)
        dataset["Embarked_C"] = dataset["Embarked"].map(lambda x: 1 if x=="C" else 0)
        dataset["Embarked_Q"] = dataset["Embarked"].map(lambda x: 1 if x=="Q" else 0)
        dataset["Embarked_S"] = dataset["Embarked"].map(lambda x: 1 if x=="S" else 0)
        dataset["Embarked_Unknown"] = dataset["Embarked"].map(lambda x: 1 if x is np.nan else 0)
        
        for columns in self.numerical_cols:
            dataset[columns] = dataset[columns].fillna(value=dataset[columns].median())
        
        dataset = dataset.drop(["Name","Ticket","PassengerId","Embarked","Sex"],axis=1)
        
        return dataset
    
# Evaluation with cross_val_score
my_pipeline = Pipeline(steps=[('preprocessor',CustomTransformer()),
                             ('model', RandomForestClassifier(random_state=0,n_estimators=200, max_depth=5))])

score = cross_val_score(my_pipeline, X, y, cv=5, scoring="accuracy")

print("Le score est de {:.3f}".format(score.mean()))

""" Prediction des survivants """

custom_transformer = CustomTransformer()

model = RandomForestClassifier(random_state=0,n_estimators=200, max_depth=5)

model.fit(custom_transformer.fit(X).transform(X),y)

test_data=pd.read_csv("test.csv")

X_test = custom_transformer.transform(test_data)

test_predict = model.predict(X_test)

temp_dict = {"PassengerId":test_data.PassengerId.tolist(),"Survived":test_predict.tolist()}

submission = pd.DataFrame.from_dict(temp_dict)
submission.Survived = submission.Survived.astype('int')

submission.to_csv("gender_submission_random_forest.csv",index=False)
