import pandas as pd
import numpy as np

df = pd.read_csv('Activity_Master_Data_Labelled.csv')

size = 5800
replace = False

fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
validation_data = df.groupby('Activity', as_index=False).apply(fn).reset_index(drop=True)

training_data = df.drop(validation_data.index)

training_data.to_csv('training_data.csv', sep=',')
validation_data.to_csv('validation_data.csv', sep=',')