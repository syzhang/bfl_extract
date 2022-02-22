"""
compare pain vs no pain with questionnaire data cross validation
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import optuna

from compare_hyperparams import load_feats, full_labels, remove_subjs, feature_importance, cv_classify
from compare_bfl_qsidp import proc_qsidp, load_qscode, combinations_all

def make_data_painquestion(bestIC, qs_ls='all', idp_ls='all'):
    bfloutput_dir='/well/tracey/shared/fps-ukb/bigflica_output/output_painquestion_500/'
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    d = f'Result_IC{bestIC}'
    data_dir = os.path.join(bfloutput_dir, d)
    # load labels
    df_label = pd.read_csv(os.path.join(curr_dir, 'labels_full', 'label_painquestion.csv'))
    # load qsidp (section to match bfl, impute, dummify)
    if bestIC==0:
        df_qsidp = pd.read_csv(os.path.join(curr_dir,'qsidp','qsidp_painquestion.csv'))
        df_qs_imputed_dum = proc_qsidp(df_qsidp, df_label, questionnaire=qs_ls, idp=idp_ls) # hack
        df_bfl_qsidp = df_label.merge(df_qs_imputed_dum, left_on='eid', right_on='eid', how='left',indicator=False)
    else:
        df_featout_ex = remove_subjs(data_dir, df_label, remove_dup=False)
        if qs_ls is not None or idp_ls is not None:
            df_qsidp = pd.read_csv(os.path.join(curr_dir,'qsidp','qsidp_painquestion.csv'))
            df_qs_imputed_dum = proc_qsidp(df_qsidp, df_featout_ex, questionnaire=qs_ls, idp=idp_ls)
            print(f'df_qs_imputed_dum shape={df_qs_imputed_dum.shape}')
            # merge bfl and qsidp
            df_bfl_qsidp = df_featout_ex.merge(df_qs_imputed_dum, left_on='eid', right_on='eid', how='left',indicator=False)
        else:
            df_bfl_qsidp = df_featout_ex
            
    print(f'df_bfl_qsidp shape={df_bfl_qsidp.shape}')
    return df_bfl_qsidp


def objective(trial, X, y):
    """tuning using optuna"""
    from sklearn.model_selection import train_test_split
    import sklearn.ensemble
    from sklearn.model_selection import cross_validate
    
    rf_max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    list_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    rf_n_estimators = trial.suggest_categorical('n_estimators', list_trees)
    rf_max_features = trial.suggest_uniform('max_features', 0.15, 1.0)
    rf_min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
    rf_min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 14)
    rf_max_samples = trial.suggest_uniform('max_samples', 0.6, 0.99)
            
    classifier_obj = sklearn.ensemble.RandomForestClassifier(
        max_depth=rf_max_depth, n_estimators=rf_n_estimators, max_features=rf_max_features, 
        min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf, max_samples=rf_max_samples,
        bootstrap=True, verbose=0
    )

    # balanced accuracy mean
    scoring = {'acc': 'balanced_accuracy',
               'roc_auc': 'roc_auc'}
    scores = cross_validate(classifier_obj, X, y, scoring=scoring,
                             cv=4, return_train_score=False)
    return np.mean(scores['test_acc']), np.mean(scores['test_roc_auc'])


def fit_bp(bestIC, qs_ls, idp_ls, feat_scaler=True, feat_balance=True, fit_n=30):
    """fit best params using optuna"""
    # load bfl
    df_bfl_qsidp = make_data_painquestion(bestIC, qs_ls=qs_ls, idp_ls=idp_ls)
    # scale balance
    X_train, y_train = load_feats(df_bfl_qsidp, test_size=0.25, dummies=False,
                                  train=False, balance=feat_balance, scaler=feat_scaler)
    print(X_train.shape, y_train.shape)
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=fit_n)
    return study

def cv_loop(bestIC, qs_ls, idp_ls, feat_scaler=True, feat_balance=True, fit_n=10):
    """cv loop of all possible input combinations"""
    # fit bp
    study_bp = fit_bp(bestIC, qs_ls, idp_ls, feat_scaler, feat_balance, fit_n)
    # cv results
    df_cv = sum_results(study_bp)
    return df_cv

def sum_results(study_bp):
    """saving optuna results to df"""
    bps = study_bp.best_trials[0].params
    bps['balanced_accuracy'] = study_bp.best_trials[0].values[0]
    bps['roc_auc'] = study_bp.best_trials[0].values[1]
    bps_dct = {k:[v] for k,v in bps.items()}
    df_out = pd.DataFrame.from_dict(bps_dct)
    return df_out

def regroup_ls(feat_in, 
               qs_all=['cognitive','demographic','lifestyle','mental'], 
               idp_all=['t1vols','subcorticalvol','fast','t2star','wdmri','taskfmri']):
    """regroup feature inputs into qs and idp"""
    qs_in, idp_in = [], []
    for i in feat_in:
        if i in qs_all:
            qs_in.append(i)
        elif i in idp_all:
            idp_in.append(i)
    return qs_in, idp_in

# run
if __name__=="__main__":
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    
    # loop through
    IC = int(sys.argv[1]) # to split up work to different qsub

    add_ls = str(sys.argv[2]) # to split up work to different qsub
    qs_all = ['cognitive','demographic','lifestyle','mental']
    idp_all = ['t1vols','subcorticalvol','fast','t2star','wdmri','taskfmri']#'dmri','t2weighted',

    if add_ls == 'qs':
        added_ls = combinations_all(qs_all)
        idp_in = None
    elif add_ls == 'idp':
        added_ls = combinations_all(idp_all)
        qs_in = None
    elif add_ls == 'qsidp':
        add_all = ['t2star', 'taskfmri', 'lifestyle', 'mental']
        added_ls = combinations_all(add_all)
    else: #using qsidp without bfl output
        add_all = qs_all + idp_all
        added_ls = combinations_all(add_all)
    
    cv_res = []

    for feat in added_ls:
        # clean up empty qs
        if len(list(feat))==0:
            feat_in = None
        else:
            feat_in = list(feat)
            
        if add_ls == 'qs':
            qs_in = feat_in
        elif add_ls == 'idp':
            idp_in = feat_in
        elif add_ls == 'qsidp':
            if feat_in != None:
                qs_in, idp_in = regroup_ls(feat_in)
            else:
                qs_in, idp_in = None, None
        else: # qs and idp only, no bfl
            IC = 0
            print(feat_in)
            if feat_in != None:
                qs_in, idp_in = regroup_ls(feat_in)
            else:
                continue


        print(f'Currently running - bestIC={IC}, qs_ls={qs_in}, idp_ls={idp_in}')
        df_cv = cv_loop(IC, qs_in, idp_in, fit_n=15)
        df_cv['bestIC'] = IC
        df_cv['qsidp'] = str(feat_in)
        cv_res.append(df_cv)
                
    df_save = pd.concat(cv_res)
    df_save.to_csv(os.path.join(curr_dir, 'cv_results', 'painquestion', f'cv_results_{IC}IC_{add_ls}.csv'), index=None)
    
    