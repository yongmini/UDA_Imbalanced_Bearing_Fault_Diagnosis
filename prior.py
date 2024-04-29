import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from tsfresh.feature_extraction import feature_calculators as fc

def get_prior_features(X):
 
  
    def calculate_stat_feature(x):

        
        mean = np.mean(x, axis=1)
        absolute_mean = np.mean(np.abs(x), axis=1)
        std = np.std(x, axis=1)
        variance = np.var(x, axis=1)
        rms = np.sqrt(np.mean(x**2, axis=1))
        absolute_max = np.max(np.abs(x), axis=1)
   
        skewness = skew(x,axis=1)
        kurtosis_val = kurtosis(x, axis=1)
        crest_factor = absolute_max / rms
        margin_factor = absolute_max / variance
        shape_factor = rms / absolute_mean
        impulse_factor = absolute_max / absolute_mean
        peak_index= absolute_max / (std*variance)
        kurtosis_index= (kurtosis_val*crest_factor) / std
        
    #---------------------------------------------------------------------------
        mean_col = pd.DataFrame(mean, columns=['MA'])
        std_col = pd.DataFrame(std, columns=['SD'])
        var_col = pd.DataFrame(variance, columns=['VA'])
        rms_col = pd.DataFrame(rms, columns=['RM'])
        absolute_max_col = pd.DataFrame(absolute_max, columns=['AM'])
        skewness_col = pd.DataFrame(skewness, columns=['SI'])
        kurtosis_val_col = pd.DataFrame(kurtosis_val, columns=['KT'])
        crest_factor_col = pd.DataFrame(crest_factor, columns=['CF'])
        margin_factor_col = pd.DataFrame(margin_factor, columns=['MF'])
        shape_factor_col = pd.DataFrame(shape_factor, columns=['SF'])
        impulse_factor_col = pd.DataFrame(impulse_factor, columns=['IF'])
   
        peak_index_col = pd.DataFrame(peak_index, columns=['PI'])
        kurtosis_index_col = pd.DataFrame(kurtosis_index, columns=['KI'])
   

        df = pd.concat([
            mean_col, std_col, var_col, rms_col, absolute_max_col,
            skewness_col, kurtosis_val_col, crest_factor_col,
            margin_factor_col, shape_factor_col, impulse_factor_col,
            peak_index_col,kurtosis_index_col
        ], axis=1)
         
         

        
        
        return df


    def calculate_time_feature(x):


        mean_abs_change = np.zeros((x.shape[0],))
        zero_crossing_rate = np.zeros((x.shape[0],))
        autocorrelation10 = np.zeros((x.shape[0],))
        
        
        for i in range(x.shape[0]):
  
            zero_crossing_rate[i] = np.sum(np.abs(np.diff(np.sign(np.squeeze(x)[i])))) / 2.0
            mean_abs_change[i] = fc.mean_abs_change(np.squeeze(x)[i])
            autocorrelation10[i] = fc.autocorrelation(np.squeeze(x)[i],10)
        df = pd.DataFrame({
        'mac': mean_abs_change,
        'zcr': zero_crossing_rate,
        #'autocorrelation10': autocorrelation10,
        })
        
        return df

    
    stat_features=calculate_stat_feature(X)
    time=calculate_time_feature(X)
    prior_features= pd.concat([stat_features,time], axis=1)

    return prior_features

