import numpy as np
import pandas as pd
from pipeline import G2JN_Pipeline



def load_boston():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    boston_dataset = pd.read_csv("data/housing.csv", header=None, delimiter=r"\s+", names=column_names)
    X_bos, y_bos = boston_dataset.iloc[:,:-1], boston_dataset.iloc[:,-1]
    name = "Boston-Housing"
    return  X_bos, y_bos, name

def load_motors():
  motors_dataset = pd.read_csv('data/freMTPL2freq.csv')#.sample(frac=0.1,random_state=1).reset_index(drop=True)
  motors_dataset['Frequency'] = motors_dataset['ClaimNb'] / motors_dataset['Exposure']
  X_mot, y_mot = motors_dataset.drop(['IDpol', 'ClaimNb', 'Exposure', 'Frequency'],axis=1), motors_dataset['Frequency']
  categorical_columns = X_mot.dtypes[X_mot.dtypes == 'object'].index
  # Perform one-hot encoding on the categorical columns
  one_hot_df = pd.get_dummies(X_mot[categorical_columns], dtype=float)
  X_mot =  (X_mot.drop(categorical_columns,axis=1)).join(one_hot_df).reset_index(drop=True)
  name = "French-Motor-claims"
  return X_mot, y_mot,name

def parameters_tuning(X,y,name):
  params_df = pd.DataFrame()
  res = G2JN_Pipeline(X, y, name)
  res.fit(conf_int =95)
  for thres in [0.5,0.7]:
    for percenti_t in [25,50,75]:
      for f_thr in [10,30,95]:
        for min_in_bin in [5,10,15]:
          for mutate in [True,False]:
            if mutate:
              for frac in [0.05,0.2,0.5]:
                try:
                  res.transform( samples_per_bin =30,
                                        max_bins = 750,
                                        method='mean',
                                        threshold=thres,
                                        percentile_threshold = percenti_t,
                                        min_amount_samples_in_bin = min_in_bin,
                                        mutate =mutate,
                                        frac = frac,
                                        f_thr = f_thr)  
                  params_df = pd.concat([params_df,pd.DataFrame(res.parameters,index=[0])]).reset_index(drop=True)
                  params_df.to_csv('motors_tuning_ALL_CTGAN.csv',index=False)
                except:
                  continue
            else:
              try:
                res.transform( samples_per_bin =30,
                                      max_bins = 750,
                                      method='mean',
                                      threshold=thres,
                                      percentile_threshold = percenti_t,
                                      min_amount_samples_in_bin = min_in_bin,
                                      mutate =mutate,
                                      frac = frac,
                                      f_thr = f_thr)  
                params_df = pd.concat([params_df,pd.DataFrame(res.parameters,index=[0])]).reset_index(drop=True)
                params_df.to_csv('motors_tuning_ALL_CTGAN.csv',index=False)
              except:
                continue
  return 

if __name__ == "__main__":
    #parameters_tuning(load_motors())
    X_bos, y_bos, name = load_boston()
    print("--------------------------------"*3)
    print("--------------------------------"*3)
    print("--------------------------------"*3)
    pipline_bos = G2JN_Pipeline(X_bos, y_bos, name)
    pipline_bos.fit()
    pipline_bos.transform()
    print("--------------------------------"*3)
    print("--------------------------------"*3)
    print("--------------------------------"*3)
    X_mot, y_mot, name = load_motors()
    pipline_mot = G2JN_Pipeline(X_mot, y_mot, name )
    pipline_mot.fit()
    pipline_mot.transform(  max_bins = 750,
                            threshold=0.7,
                            min_amount_samples_in_bin = 15,
                            mutate =True,
                            frac = 0.2,
                            f_thr = 95)

