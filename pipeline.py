import numpy as np
import pandas as pd
from utils.help_func import *
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

class G2JN_Pipeline:
    def __init__(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        name,
        xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror'),

        ):
        if not isinstance(X, pd.DataFrame):
            raise Exception("Train dataset must be of type 'pandas.core.frame.DataFrame'")
        if not isinstance(y, pd.Series):
            raise Exception(f"The target column must be of type 'pandas.core.series.Series'")
        self.X = X
        self.y = y
        self.columns = X.columns
        self.y_name = y.name
        self.name = name
        self.xg_reg = xg_reg # baseline model - xgboost
        self.parameters ={
          'name':self.name,
          'org_rmse': None,
          'imprv_rmse':None,
          'conf_int': 0,
          'samples_per_bin': 0,
          'max_bins': 0,
          'bin_method': 0,
          'threshold':0,
          'percentile_threshold': 0,
          'min_amount_samples_in_bin' : 0,
          'mutate': 0,
          'mutate_frac':0}

        
    def fit(self,conf_int = 95,mac_seed= 101):
        print(f"Initializing Pipeline on {self.name} dataset...\n")
        # Data split
        self.splited = split_bin(self.X,self.y)
        self.parameters['conf_int']= conf_int
        # Initial prediction
        self.xg_reg.fit(self.splited['X_train'],self.splited['y_train'])
        self.preds = self.xg_reg.predict(self.splited['X_test'])
        self.rmse_org = np.sqrt(mean_squared_error(self.splited['y_test'], self.preds))
        self.parameters['org_rmse'] = self.rmse_org
        self.mae_org = mean_absolute_error(self.splited['y_test'], self.preds)
        self.parameters['org_mae'] = self.mae_org
        print(f"Initial RMSE: {round(self.rmse_org,2):.2f}\n")
        print(f"Initial MAE: {round(self.mae_org,2):.2f}\n")

        
        # Apply Macest to get prediction interval
        conf_interval = get_conf_interval(conf_int, self.splited,mac_seed)

        self.conf_df = pd.DataFrame()
        self.conf_df['lower_bound_pred'] = conf_interval.squeeze()[:,0]
        self.conf_df['upper_bound_pred'] = conf_interval.squeeze()[:,1]
        self.conf_df['diff_prop'] = abs(self.conf_df['upper_bound_pred'] - self.conf_df['lower_bound_pred'])
        self.conf_df['y_pred'] = self.preds
        self.conf_df = self.conf_df.query("lower_bound_pred<=y_pred <= upper_bound_pred").reset_index(drop=True)  # Drop faulted samples
        print(self.conf_df.head(10).to_string(),"\n")


    def transform(
        self,
        samples_per_bin=30,
        max_bins=1000,
        method="mean",
        threshold=0.5,
        percentile_threshold = 25,
        min_amount_samples_in_bin = 10,
        gen_seed = 1,
        gen_len = 1,
        mutate = False,
        frac = 0.05,
        min_rep = 5,
        f_thr = False

        ):
        self.parameters['name']=self.name
        self.parameters['Gen_Model'] ='TVAE'
        self.parameters['samples_per_bin'] = samples_per_bin
        self.parameters['max_bins'] = max_bins
        self.parameters['bin_method']=method
        self.parameters['threshold']=threshold
        self.parameters['percentile_threshold'] = percentile_threshold
        self.parameters['min_amount_samples_in_bin'] = min_amount_samples_in_bin
        self.parameters['mutate'] = mutate
        self.parameters['frac']=frac
        self.parameters['min_rep']= min_rep
        self.parameters['f_thr'] = f_thr
        

        # Bin the data
        self.bins_df, _ = assign_bins( # CHECK IF self.splited GETS BINNED ALONG THE WAY TOO... IT DOES ON COLAB BUT NOT necessarily ON CLASS
            self.splited['X_train'],
            self.splited['y_train'],
            samples_per_bin,
            max_bins,
            method,
            threshold) 
        
        # Analyze bin uncertainty
        binslist = np.append(self.bins_df['start_bin_value'].values, self.bins_df['end_bin_value'].iloc[-1])
        self.conf_df['bins']= pd.cut(self.conf_df['y_pred'] , bins = binslist,labels = self.bins_df.index.tolist())
        diff_per_bin = pd.DataFrame(self.conf_df.groupby('bins')['diff_prop'].mean()).reset_index()
        top_percentile = np.percentile(self.conf_df.groupby('bins')['diff_prop'].mean().dropna(), percentile_threshold)
        uncertainty_bins = diff_per_bin[diff_per_bin.diff_prop > top_percentile]['bins']

        # Assign Epistemic and Aleatoric uncertainties
            # Epistemic uncertainty
        self.bins_df.loc[(self.bins_df.suspected_low_in_data_bins == True) &
         (self.bins_df.index.isin(uncertainty_bins) ) & (self.bins_df['count_samples'] > 1) ,'uncertainty_type'] = 'Epistemic'

            # Another Epistemic uncertainty - if there are bins that didnt show up on the conf_df - suspect as epistemic
        if len(diff_per_bin[diff_per_bin['diff_prop'].isna()].bins.tolist()) > 0:
            self.bins_df.loc[(self.bins_df.suspected_low_in_data_bins == True) &
             (self.bins_df['count_samples'] > min_amount_samples_in_bin) &
              (self.bins_df.index.isin(diff_per_bin[diff_per_bin['diff_prop'].isna()].bins.tolist()) ),'uncertainty_type'] = 'Epistemic'

            # Aleatoric uncertainty
        self.bins_df.loc[(self.bins_df.suspected_low_in_data_bins == False) & (self.bins_df.index.isin(uncertainty_bins) ),'uncertainty_type'] = 'Aleatoric'

        print((self.bins_df.sort_values(by = "count_samples",ascending=False).head(50)).to_string(),"\n") #beautify pandas df print
        
        data = pd.concat([self.splited['X_train'],self.splited['y_train']], axis=1).reset_index(drop=True)
        # Apply MEDIAN MUTATION to deal with aleatoric uncertainty - each bin is dealy separately and then re-stack all data together

        if mutate:
          print("Mutation Occurred!")
          data_no_alea = data[data['bins'].isin(self.bins_df[self.bins_df['uncertainty_type'] != 'Aleatoric'].index)].reset_index(drop=True)
          data_no_ol = pd.concat([median_mutation(data[data['bins'] == val],frac,min_rep) for val in self.bins_df[self.bins_df['uncertainty_type'] == 'Aleatoric'].index])
          #data = pd.concat([data_no_alea,data_no_ol]).reset_index(drop=True)
          data = pd.concat([data_no_alea,data_no_ol]).reset_index(drop=True)
        # Remove outliers from each bin separately and then re-stack all data together

        # Remove outliers from each bin separately and then re-stack all data together
        print(f"Total number of samples in bins **before** outliers removal: {data.shape[0]}")
        data_no_alea = data[data['bins'].isin(self.bins_df[self.bins_df['uncertainty_type'] != 'Aleatoric'].index)].reset_index(drop=True)
        data_no_ol = pd.concat([remove_outliers(data[data['bins'] == val]) for val in self.bins_df[self.bins_df['uncertainty_type'] == 'Aleatoric'].index])
        data_no_ol = pd.concat([data_no_alea,data_no_ol]).reset_index(drop=True)
       # data_no_ol = data_no_ol.sample(frac=1).reset_index(drop=True) # Shuffle the new data
        print(f"Total number of samples in bins **after** outliers removal: {data_no_ol.shape[0]}")

        # Generate data for uncertainties
        regenerated_data = pd.concat([generate_data(data_no_ol[data_no_ol['bins'] == val],n_gen=gen_len,seed=gen_seed) for val in self.bins_df[self.bins_df['uncertainty_type'] == 'Epistemic'].index])
        data_new = pd.concat([data_no_ol,regenerated_data]).reset_index(drop=True)
        #data_new = data_new.sample(frac=1).reset_index(drop=True) # Shuffle the new data
        self.data_new = data_new
        print(f"Total number of samples in new generated data: {data_new.shape[0]}")

        # Insert last pipeline of feature selection
        last_pipeline = Pipeline([
        ('feature_selection', SelectPercentile(score_func=f_regression, percentile=f_thr)),
        ('model', self.xg_reg)])
        
        if f_thr == False:
          last_pipeline = Pipeline([
                ('model', self.xg_reg)])

        # Final prediction
        last_pipeline.fit(self.data_new.drop(labels= [self.y_name,'bins'],axis=1),self.data_new.filter([self.y_name]))
        preds_new = last_pipeline.predict(self.splited['X_test'])
        self.rmse_imprv = np.sqrt(mean_squared_error(self.splited['y_test'], preds_new))
        self.parameters['imprv_rmse'] = self.rmse_imprv
        self.mae_imprv = mean_absolute_error(self.splited['y_test'], preds_new)
        self.parameters['imprv_mae'] =  self.mae_imprv
        print("-------------------------------------------------")   
        print(f"Initial RMSE: {round(self.rmse_org,2):.2f}")
        print(f"Improved RMSE: {round(self.rmse_imprv,2):.2f}")
        if ((self.rmse_imprv - self.rmse_org) / self.rmse_org) <0:
          self.rate = round(100*abs((self.rmse_imprv - self.rmse_org) / self.rmse_org),2)
          print(f"\nRMSE improvement rate: {round(self.rate,2):.2f}%")
        print("-------------------------------------------------")    
        print(f"Initial MAE: {round(self.mae_org,2):.2f}")
        print(f"Improved MAE: {round(self.mae_imprv,2):.2f}")   
        if ((self.mae_imprv - self.mae_org) / self.mae_org) <0:
          self.mae_rate = round(100*abs((self.mae_imprv - self.mae_org) / self.mae_org),2)
          print(f"\nMAE Improvement rate: {round(self.mae_rate,2):.2f}%")




