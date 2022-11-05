import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('Datos/train_set.csv')
train_df = train_df.dropna()

features = train_df[['duration', 'start_lat', 'start_lon',
                     'end_lat', 'end_lon', 'trip_route_category',
                     'start_station', 'end_station']]

target = train_df['passholder_type']

trip = pd.get_dummies(features['trip_route_category'], prefix='Trip_category')
features_dummies = pd.concat([features, trip], axis=1)
features_dummies = features_dummies.drop(['trip_route_category'], axis = 1)
features_dummies.reset_index(inplace=True, drop=True)

x_train, x_val, y_train, y_val = train_test_split(features_dummies, target, test_size=0.3)

tree_model = DecisionTreeClassifier(criterion='entropy',
                                    min_samples_split=20,
                                    min_samples_leaf=5,
                                    max_depth=15)

tree_model.fit(x_train, y_train)


def model_prediction(data: pd.DataFrame):
    target_predict = tree_model.predict(data)
    results = pd.Series(target_predict)
    return results


train_score = tree_model.score(x_train, y_train) * 100
print(f'Training score = {train_score:.2f}%')
validation_score = tree_model.score(x_val, y_val) * 100
print(f'Validation score = {validation_score:.2f}%')
