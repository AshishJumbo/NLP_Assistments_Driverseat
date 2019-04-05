import pandas as pd

df_root = pd.read_csv('training_set.csv')

df_training = df_root.loc[df_root['dataset'] == 'evaluation']
print(df_root.loc[(df_root['problem_id'] == 14) & (df_root['dataset'] == 'evaluation')]['raw_answer_text'])
print(df_training.shape)

df_result = pd.read_csv('result.csv')
print(df_result.shape)

