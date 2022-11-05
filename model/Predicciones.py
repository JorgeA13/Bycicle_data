from model.Modelo_entrenamiento import model_prediction
import pandas as pd

test_df = pd.read_csv('Datos/test_set.csv')
test_df = test_df.dropna()
test_df.reset_index(inplace=True, drop=True)

test_features = test_df[['duration', 'start_lat', 'start_lon',
                         'end_lat', 'end_lon', 'trip_route_category',
                         'start_station', 'end_station']]

trip = pd.get_dummies(test_features['trip_route_category'], prefix='Trip_category')

test_dummies = pd.concat([test_features, trip], axis=1)
test_dummies = test_dummies.drop(['trip_route_category'], axis=1)

results = model_prediction(test_dummies)
results = pd.Series(results, name='passholder_type')
results_csv = pd.concat([test_df['trip_id'], results], axis=1)
results_csv.to_csv('model/Resultados.csv', index=False)
