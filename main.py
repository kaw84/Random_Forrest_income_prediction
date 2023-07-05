def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#importing data
income_data = pd.read_csv('income.csv', header = 0, delimiter = ", ")

#reassigning data
income_data['sex-int'] = income_data['sex'].apply(lambda row: 0 if row == 'Male' else 1)

income_data['country-int'] = income_data['native-country'].apply(lambda row: 0 if row == 'United-States' else 1)

#selecting a column
labels = income_data['income']
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int']]

#splitting the data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

#building the forest
forest = RandomForestClassifier(random_state = 1)

#fit the model
forest.fit(train_data, train_labels)

#checking accuracy
print(forest.score(test_data, test_labels))