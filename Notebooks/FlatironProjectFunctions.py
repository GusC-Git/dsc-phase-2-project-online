# -*- coding: utf-8 -*-
""" Contains functions that are used throughout this project code, both in cleaning and other areas.
Also contains functions used in building a model and checking that the model meets the assumptions of linear regression. Created in a collaborative effort during study group on Topic 19 in Flatiron School with our instructor, Rafael Carrasco.
"""
import numpy as np
import statsmodels.api as sm
import pandas as pd
import scipy.stats as scs

# Create a function to build a statsmodels ols model
def build_sm_ols(df, features_to_use, target, add_constant=False, show_summary=True):
    """Will build an Ordinary Least Squares model using the statsmodels package.
    
    df : a pandas dataframe that has the features you plan on using
    ---
    features_to_use : list of independent variables within df that you will use to predict your dependent variable.
    ---
    target : string object containing the feature within df that you are trying to predict.
    ---
    add_constant : boolean statement. will add a constant column to df. default to False.
    ---
    show_summary : boolean statement. will print OLS summary for extra info. default to False"""
    X = df[features_to_use]
    if add_constant:
        X = sm.add_constant(X)
    y = df[target]
    ols = sm.OLS(y, X).fit()
    if show_summary:
        print(ols.summary())
    return ols


# create a function to check the validity of your model
# it should measure multicollinearity using vif of features
# it should test the normality of your residuals 
# it should plot residuals against an xaxis to check for homoskedacity
# it should implement the Breusch Pagan Test for Heteroskedasticity
##  Ho: the variance is constant            
##  Ha: the variance is not constant


# assumptions of ols
# residuals are normally distributed
def check_residuals_normal(ols):
    residuals = ols.resid
    t, p = scs.shapiro(residuals)
    if p <= 0.05:
        return False
    return True


# residuals are homoskedasticitous
def check_residuals_homoskedasticity(ols):
    import statsmodels.stats.api as sms
    resid = ols.resid
    exog = ols.model.exog
    lg, p, f, fp = sms.het_breuschpagan(resid=resid, exog_het=exog)
    if p >= 0.05:
        return True
    return False




def check_vif(df, features_to_use, target_feature, add_constant=False, show_summary=False):
    """Will output the vif of a target feature given a feature space and dataframe.
    
    df : pandas dataframe  of the data
    ---
    features_to_use : list of independent variables within df that you will use to predict your dependent variable.
    ---
    target_feature : string object containing the feature within df that you are trying to predict. 
    """
    ols = build_sm_ols(df=df, features_to_use=features_to_use, target=target_feature, add_constant=add_constant, show_summary=show_summary)
    r2 = ols.rsquared
    return 1 / (1 - r2)
    
    
    
# no multicollinearity in our feature space
def check_vif_feature_space(df, features_to_use, vif_threshold=3.0, add_constant=False, show_summary=False):
    """Will check the vif of a feature space. If the vif surpasses the vif threshold then there is multicollinearity in the model. The funtion will print which functions are the ones that have surpassed the threshold and will return a boolean statement dependent on whether our feature space is usable or not.
    
    df : pandas dataframe  of the data
    ---
    features_to_use : list of independent variables within df that you will use to predict your dependent variable.
    ---
    vif_threshold : float. number to indicate what level of vif we are okay with. The lower the number the more stringent the requirement. default to 3.0
    """
    
    all_good_vif = True
    for feature in features_to_use:
        target_feature = feature
        _features_to_use = [f for f in features_to_use if f!=target_feature]
        vif = check_vif(df=df, features_to_use=_features_to_use, target_feature=target_feature, add_constant=add_constant, show_summary=show_summary)
        if vif >= vif_threshold:
            print(f"{target_feature} surpassed threshold with vif={vif}")
            all_good_vif = False
    return all_good_vif
        
        


def check_model(df, 
                features_to_use, 
                target_col, 
                add_constant=False, 
                show_summary=False, 
                vif_threshold=3.0):
    has_multicollinearity = check_vif_feature_space(df=df, 
                                                    features_to_use=features_to_use, 
                                                    vif_threshold=vif_threshold, add_constant=add_constant)
    if not has_multicollinearity:
        print("Model contains multicollinear features")
    
    # build model 
    ols = build_sm_ols(df=df, features_to_use=features_to_use, 
                       target=target_col, add_constant=add_constant, 
                       show_summary=show_summary)
    
    # check residuals
    resids_are_norm = check_residuals_normal(ols)
    resids_are_homo = check_residuals_homoskedasticity(ols)
    
    if not resids_are_norm or not resids_are_homo:
        print("Residuals failed test/tests")
    return ols

def check_deviation_range(df, col):
    """
    Takes in a dataframe and a target column and returns a tuple of the range of 
    the standard deviation for the column of the dataframe
    
    df : pandas dataframe
    ---
    col : string. name of column within dataframe
    """
    return (df[col].mean() - df[col].std()*3, df[col].mean() + df[col].std()*3)

def get_trimmed_dataframe(df, target_features):
    """This function will get the standard deviations of a dataframe for columns of your choice and
    will return a dataframe with all of the values within 3 standard deviations of each
    original column. The index will be reset in each one.
    
    df : pandas dataframe
    ---
    target_features : list of strings containing the features you wish to cut out the outliers from
    ---
    return output : Will be a new dataframe with a reset index. All the data related to each of the
    target features will be within 3 standard deviations of the mean.
    """
    trimmed_df = df
    for feature in target_features:
        lower, upper = check_deviation_range(df, feature)
        trimmed_df = trimmed_df[(df[feature] > lower) & (df[feature] < upper)]
    trimmed_df = trimmed_df.reset_index().drop(columns='index')
    return trimmed_df