import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Chemin du fichier train.csv", required=True)
parser.add_argument("--test", help="Chemin du fichier test.csv", required=True)
parser.add_argument("--output", help="Chemin du fichier de soumission au format csv.", required=True)

args = parser.parse_args()


# Preprocessing with custom transformer
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(CustomTransformer, self).__init__()
        self.embarked_values = ["S", "C", "Q", "Unknown"]
        self.numerical_cols = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Fare_Per_Person']

    def fit(self, data_x, data_y=None):
        return self

    def rename(self, x):
        if x in ["Capt.", "Col.", "Don.", "Dr.", "Jonkheer.", "Lady.", "Major.", "Rev.", "Sir.", "Countess."]:
            return "Aristocratic."
        if x in ["Ms.", "Mme."]:
            return "Mrs."
        if x == "Mlle.":
            return "Miss."
        return x

    def transform(self, data_x, data_y=None):
        dataset = data_x.copy()
        dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch']
        dataset['Fare_Per_Person'] = dataset['Fare'] / (dataset['Family_Size'] + 1)
        dataset["Male"] = dataset["Sex"].map(lambda x: 1 if x == "male" else 0)
        dataset["Female"] = dataset["Sex"].map(lambda x: 1 if x == "female" else 0)
        dataset["Cabin"] = dataset["Cabin"].map(lambda x: 0 if x is np.nan else 1)
        dataset["Embarked_C"] = dataset["Embarked"].map(lambda x: 1 if x == "C" else 0)
        dataset["Embarked_Q"] = dataset["Embarked"].map(lambda x: 1 if x == "Q" else 0)
        dataset["Embarked_S"] = dataset["Embarked"].map(lambda x: 1 if x == "S" else 0)
        dataset["Embarked_Unknown"] = dataset["Embarked"].map(lambda x: 1 if x is np.nan else 0)
        dataset["Title"] = dataset["Name"].map(lambda x: re.findall(r"\w+\.", x)[0])
        dataset["Title"] = dataset["Title"].map(self.rename)

        for title in dataset.Title.unique().tolist():
            dataset.loc[(dataset.Title == title), 'Age'] = dataset['Age'].apply(
                lambda x: dataset.loc[(dataset.Title == title), 'Age'].median() if pd.isnull(x) else x)

        for columns in self.numerical_cols:
            dataset[columns] = dataset[columns].fillna(value=dataset[columns].median())

        dataset = dataset.drop(["Name", "Ticket", "PassengerId", "Embarked", "Sex", "Title"], axis=1)

        return dataset


def get_accuracy(data, model=RandomForestClassifier(random_state=0, n_estimators=200, max_depth=5)):
    """ Obtenir la précision du modèle (cross validation) """

    data_y = data.Survived
    data_x = data.drop(["Survived"], axis=1)

    # Evaluation with cross_val_score
    my_pipeline = Pipeline(steps=[('preprocessor', CustomTransformer()),
                                  ('model', model)])

    cv_score = cross_val_score(my_pipeline, data_x, data_y, cv=5, scoring="accuracy")

    return cv_score.mean()


def model_transformer_fit(data, model=RandomForestClassifier(random_state=0, n_estimators=200, max_depth=5)):
    """ Preparation des données d'entrainement, fit du transformer et du modèle """

    data_y = data.Survived
    data_x = data.drop(["Survived"], axis=1)
    custom_transformer = CustomTransformer()
    model.fit(custom_transformer.fit(data_x).transform(data_x), data_y)

    return model, custom_transformer


def predict(data, model, custom_transformer, output):
    """ Prediction sur les données de test et enregistrement dans un csv """

    data_prepared = custom_transformer.transform(data)
    
    test_predict = model.predict(data_prepared)
    
    temp_dict = {"PassengerId": data.PassengerId.tolist(), "Survived": test_predict.tolist()}
    
    submission = pd.DataFrame.from_dict(temp_dict)
    submission.Survived = submission.Survived.astype('int')
    
    submission.to_csv(output, index=False)


if __name__ == "__main__":

    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)

    model_to_used, transformer_to_used = model_transformer_fit(train_data)

    score = get_accuracy(train_data, model_to_used)
    print("Le score est de {:.3f}".format(score))

    predict(test_data, model_to_used, transformer_to_used, args.output)

