import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import numpy as np  
import pandas as pd
import torch
import xgboost as xgb
from macest.regression import models as reg_mod
from sdv.tabular import TVAE as Generator #data generation- possible packages :TVAE,CTGAN,CopulaGAN
from pyod.models.knn import KNN #outliers removal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


pd.options.mode.chained_assignment = None  # default='warn'

#Assign bins
def assign_bins(
    X,
    y,
    samples_per_bin=20,
    max_bins=1000,
    method="mean",
    threshold=0.7,
):

    """
    Function which divides the data into bins
    Args:
        X, y: the data.
        samples_per_bin:
        max_bins:
        method:
        threshold:
    Return:
       bins_df:
       X:
    """
    num_of_bins = X.shape[0] // samples_per_bin
    if num_of_bins > max_bins:
        num_of_bins = max_bins
    bin_assignment, bin_ranges = pd.cut(y, num_of_bins, labels=False, retbins=True)
    X["bins"] = bin_assignment
    bin_vals = bin_assignment.value_counts().sort_index()
    bin_vals = bin_vals.reindex(range(num_of_bins), fill_value=0)
    bin_ranges[0] = 0
    bin_ranges[-1] = bin_ranges[-1] * 10
    bins_df = pd.DataFrame(
        {
            "count_samples": bin_vals,
            "start_bin_value": bin_ranges[:-1],
            "end_bin_value": bin_ranges[1:],
        }
    )

    if method == 'median':
        bins_df["suspected_low_in_data_bins"] = bins_df["count_samples"] < (bins_df["count_samples"].median())
    else:
        bins_df["suspected_low_in_data_bins"] = bins_df["count_samples"] < (bins_df["count_samples"].mean() * threshold)
        

    return bins_df, X
# Get confidence interval with Macest
def get_conf_interval(
    desired_conf_level,
    splited_data,
    seed
):

    """
    Function gets desired train confidence and calibration sets and a confidence level.
    It returns the confidence interval of the prediction as an array
    Args:
        desired_conf_level:
        splited_data: A dictionary of the data after splited
    Return:
       conf_preds:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    rf_reg = RandomForestRegressor(
        random_state=1, n_estimators=200
    )  
    
    # Train a point prediction model
    rf_reg.fit(splited_data['X_pp_train'].values, splited_data['y_pp_train'].values)

    rf_preds = rf_reg.predict(splited_data['X_conf_train'].values)
    test_error = abs(rf_preds - splited_data['y_conf_train'].values)
    np.random.seed(seed)
    torch.manual_seed(seed)
    search_args = reg_mod.HnswGraphArgs(query_kwargs={"ef": 1500})
    bnds = reg_mod.SearchBounds(k_bounds=(5, 30))

    # Initialise and fit MACEST
    np.random.seed(seed)
    torch.manual_seed(seed)
    macest_model = reg_mod.ModelWithPredictionInterval(
        rf_reg,
        splited_data['X_conf_train'].values,
        test_error,
        error_dist="laplace",
        dist_func="linear",
        search_method_args=search_args,
    )

    optimiser_args = dict(popsize=20, disp=False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    macest_model.fit(splited_data['X_cal'].values, splited_data['y_cal'].values, param_range=bnds, optimiser_args=optimiser_args)
    # predict the intervals based on the desired_conf_level
    np.random.seed(seed)
    torch.manual_seed(seed)
    conf_preds = macest_model.predict_interval(splited_data['X_test'].values, conf_level=desired_conf_level)

    return conf_preds

# Remove outlier from bins with Aleatoric uncertainty

def remove_outliers(df):
  """
  Removes outliers using pyOD package
  Args:
      df: Pandas DataFrame where X, y are stacked together

  Return:
      Pandas DataFrame without detected outliers
  """
  ODdata = df.copy()
  detector = KNN()
  detector.fit(ODdata)
  y_pred_od = detector.predict(df)
  ODdata = ODdata[y_pred_od == 0]
  return ODdata.reset_index(drop=True)

# Generate data for bin with epistemic uncertainty
def generate_data(data_bin,n_gen=1,seed=1):
  
  """
  Generating synthetic data per a given bin of data
  Args:
      data_bin: data to augment upon
      model_tvae: the data synthesis model generator
      n_gen: number of samples to generate where 1 equals the same size of original data 

  Return:
      generated new data
  """
  model_generator = Generator()
  np.random.seed(seed)
  torch.manual_seed(seed)
  model_generator.fit(data_bin)

  np.random.seed(seed)
  torch.manual_seed(seed)
  new_data = model_generator.sample(num_rows=len(data_bin)*n_gen)

  return new_data
# Split the data into 4 groups 
def split_bin(X, y):

  """
  Splits dataset into Train, Conf,Cal and Test sets
  Args:
      X, y: Pandas.DataFrame

  Return:
      splited_data: Dictionary
  """

  X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.15, random_state=10)
  X_pp_train, X_conf_train, y_pp_train, y_conf_train = train_test_split(X_train, y_train, test_size=0.333333333333, random_state=1)
  X_conf_train, X_cal, y_conf_train, y_cal, =  train_test_split(X_conf_train, y_conf_train, test_size=0.333333333333, random_state=1)

  splited_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'X_pp_train': X_pp_train,
    'y_pp_train': y_pp_train,
    'X_conf_train': X_conf_train,
    'X_cal': X_cal,
    'y_conf_train': y_conf_train,
    'y_cal': y_cal,
  }

  return splited_data


def median_mutation(data_bin,frac = 0.05,min_rep = 5,seed=1): 
  """
  Creates a mutation for a fraction of the data. assigns replace values for bin's median
  Args:
      data_bin: a bin with aleatoric uncertainty - Pandas.DataFrame
      frac: percentage of mutation - float
      min_rep: minimum replacements for median mutation if frac of data < 1 - int

  Return:
      New mutated data bin - Pandas.DataFrame
  """
  data_mut= data_bin.copy()
  # If less than 20 samples and more than 5, insert 5 median samples. if less than 5-> double the size with median samples
  if len(data_mut)<20:
    x = data_mut.iloc[0:min_rep,:] if len(data_mut) > min_rep else data_mut.iloc[0:len(data_mut),:]
    x.iloc[:,:] = data_bin.median()
    data_mut = pd.concat([data_bin,x])
    return data_mut.sample(frac = 1,random_state=seed)#.reset_index(drop=True)

  # Mutate 5% or minimum 5 samples(default values) of the data with median replacement if there ar
  idx = data_mut.sample(frac=frac,random_state=seed).index if len(data_mut) > 100 else data_mut.sample(min_rep,random_state=42).index # get random samples to replace with median values
  data_mut.loc[idx] =data_bin.median().values
  return data_mut.sample(frac = 1,random_state=seed).reset_index(drop=True)
