import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 6})

train_df = pd.read_csv('Datos/train_set.csv')
print(train_df.info())
train_df = train_df.dropna()
print(train_df.shape)

plt.figure(figsize=(20, 15))
ax1 = sns.countplot(data=train_df, x='start_station',
              order=train_df.start_station.value_counts().iloc[:30].index)

ax1.set_title('Estaciones donde más viajes se incian (Top 30)', loc='center')

for p in ax1.patches:
    height = p.get_height()
    ax1.text(x=p.get_x()+(p.get_width()/2),
             y=height+500,
             s=f'{height:.0f}',
             ha='center')

plt.figure(figsize=(12, 8))
ax2 = sns.countplot(data=train_df, x='trip_route_category')

ax2.set_title('Diferencia entre viajes redondos y de solo 1 viaje', loc='center')

for p in ax2.patches:
    height = p.get_height()
    ax2.text(x=p.get_x()+(p.get_width()/2),
             y=height+3000,
             s=f'{height:.0f}',
             ha='center')

plt.figure(figsize=(15, 8))
ax3 = sns.countplot(data=train_df, x='trip_route_category', hue='passholder_type')

ax3.set_title('Viajes redondos y de 1 vuelta por tipo de pase', loc='center')

for p in ax3.patches:
    height = p.get_height()
    ax3.text(x=p.get_x()+(p.get_width()/2),
             y=height+3000,
             s=f'{height:.0f}',
             ha='center')

plt.figure(figsize=(12,8))
ax4 = sns.countplot(data=train_df, x='passholder_type')

ax4.set_title('Comparativa entre los tipos de pases', loc='center')

for p in ax4.patches:
    height = p.get_height()
    ax4.text(x=p.get_x()+(p.get_width()/2),
             y=height+5000,
             s=f'{height:.0f}',
             ha='center')

plt.show()
