import numpy as np
import pandas as pd
from utils.help_func import *
import xgboost as xgb
from sklearn.metrics import mean_squared_error

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

        
    def fit(self,conf_int = 95,mac_seed= 101):
        print(f"Initializing Pipeline on {self.name} dataset...\n")
        # Data split
        self.splited = split_bin(self.X,self.y)
        # Initial prediction
        self.xg_reg.fit(self.splited['X_train'],self.splited['y_train'])
        self.preds = self.xg_reg.predict(self.splited['X_test'])
        self.rmse_org = np.sqrt(mean_squared_error(self.splited['y_test'], self.preds))
        print("Initial RMSE: %f" % (self.rmse_org),"\n")
        # Apply Macest to get prediction interval
        conf_interval = get_conf_interval(conf_int, self.splited,mac_seed)

        self.conf_df = pd.DataFrame()
        self.conf_df['lower_bound_pred'] = conf_interval.squeeze()[:,0]
        self.conf_df['upper_bound_pred'] = conf_interval.squeeze()[:,1]
        self.conf_df['diff_prop'] = abs(self.conf_df['upper_bound_pred'] - self.conf_df['lower_bound_pred'])
        self.conf_df['y_pred'] = self.preds
        self.conf_df = self.conf_df.query("lower_bound_pred<=y_pred <= upper_bound_pred").reset_index(drop=True)  # Drop faulted samples


    def transform(
        self,
        samples_per_bin=30,
        max_bins=1000,
        method="mean",
        threshold=0.7,
        percentile_threshold = 25,
        min_amount_samples_in_bin = 10,
        gen_seed = 1,
        gen_len = 1,

        ):

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

        print((self.bins_df.sort_values(by = "count_samples",ascending=False).head(50)).to_string(),"\n") #beautify pandas df print?

        # Remove outliers from each bin separately and then re-stack all data together
        print(f"Total number of samples in bins **before** outliers removal: {self.splited['X_train'].shape[0]}")
        data = pd.concat([self.splited['X_train'],self.splited['y_train']], axis=1).reset_index(drop=True)
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

        # Final prediction
        self.xg_reg.fit(self.data_new.drop(labels= [self.y_name,'bins'],axis=1),self.data_new.filter([self.y_name]))
        preds_new = self.xg_reg.predict(self.splited['X_test'])
        self.rmse_imprv = np.sqrt(mean_squared_error(self.splited['y_test'], preds_new))
        print("\nInitial RMSE: %f" % (self.rmse_org))
        print("\nImproved RMSE: %f" % (self.rmse_imprv))
        if ((self.rmse_imprv - self.rmse_org) / self.rmse_org) <0:
          self.rate = round(100*abs((self.rmse_imprv - self.rmse_org) / self.rmse_org),2)
          print("\nImprovement rate: %f" % (self.rate),"%")



