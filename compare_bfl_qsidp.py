"""
pipe for combinations of bfl and qsidp data
"""

import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier

from compare_hyperparams import full_labels, remove_subjs, load_feats, cv_classify

def extract_qs(df_subjects, df_questionnaire, visits=[2]):
    """extract questionnaire set out of 5 possible"""
    # load questionnaire code of interest
    field_code = df_questionnaire['code'].to_list()
    # extract all fields with questionnaire code
    field_cols = []
    for code in field_code:
        # cols_ls = [col for col in df_subjects.columns if str(code)+'-' in col]
        code_root = str(code)+'-'
        cols_ls = [col for col in df_subjects.columns if col[:len(code_root)]==code_root]
        if visits != None: # limit to visits only
            if len(cols_ls) > 1:
                cols_exclude = []
                for visit in visits:
                    for col in cols_ls:
                        if '-'+str(visit) in col:
                            cols_exclude.append(col)
                cols_ls = cols_exclude
        else:
            cols_ls = cols_ls
        field_cols += cols_ls
    # append eid
    field_cols += ['eid']
    # remove duplicate
    field_cols_rm = list(set(field_cols))
    df_qs = df_subjects[field_cols_rm]
    # remove duplicated columns
    df_qs_rm = df_qs.loc[:, ~df_qs.columns.duplicated()]
    return df_qs_rm

def load_qscode(questionnaire='all', idp=None, curr_dir='/well/seymour/users/uhu195/python/extract_npy'):
    """load questionnaire and idp code"""
    base_dir = os.path.join(curr_dir, 'bbk_codes')
    # questionnaire data
    df_qs = pd.DataFrame()
    if questionnaire!=None and len(questionnaire)!=0:
        questionnaire_ls = ['lifestyle','mental','cognitive','digestive','cwp','demographic']
        if (questionnaire!='all') and (questionnaire in questionnaire_ls):
            df_qs = pd.read_csv(os.path.join(base_dir, questionnaire+'_code.csv'))
        elif (questionnaire!='all') and (type(questionnaire) is list): # multiple qs sets
            qs_ls = []
            for i in questionnaire:
                fname = i+'_code.csv'
                fpath = os.path.join(base_dir, fname)
                qs_ls.append(pd.read_csv(fpath))
            df_qs = pd.concat(qs_ls)
        elif questionnaire=='all':
            questionnaire_ls = ['lifestyle','mental','cognitive','demographic']
            qs_ls = []
            for qs in questionnaire_ls:
                qs_ls.append(pd.read_csv(os.path.join(base_dir,qs+'_code.csv')))
            df_qs = pd.concat(qs_ls)
        else:
            raise ValueError('Questionnaire code does not exist.')
    # idp data
    df_idp = pd.DataFrame()
    if idp!=None and len(idp)!=0:
        idp_ls = ['dmri','wdmri','fast','subcorticalvol','t1vols','t2star','t2weighted','taskfmri']
        if (idp!='all') and (idp in idp_ls): # single idp set
            df_idp = pd.read_csv(os.path.join(base_dir, 'idp_'+idp+'_code.csv'))
        elif (idp!='all') and (type(idp) is list): # multiple idp sets
            idpc_ls = []
            for i in idp:
                fname = 'idp_'+i+'_code.csv'
                fpath = os.path.join(base_dir, fname)
                idpc_ls.append(pd.read_csv(fpath))
            df_idp = pd.concat(idpc_ls)
        elif idp=='all': # all idp sets
            idpc_ls = []
            for i in idp_ls:
                fname = 'idp_'+i+'_code.csv'
                fpath = os.path.join(base_dir, fname)
                idpc_ls.append(pd.read_csv(fpath))
            df_idp = pd.concat(idpc_ls)
        else:
            raise ValueError('IDP code does not exist.')
    # combine questionnaire with idp
    df_out = pd.concat([df_qs, df_idp])
    return df_out

def impute_qs(df, nan_percent=0.9, freq_fill='median', 
              transform=False, transform_fn='sqrt'):
    """impute questionnaire df"""
    df_copy = df.copy()
    # replace prefer not to say and remove object
    df_copy = replace_noans(df_copy)
    # replace multiple choice fields
    df_copy = replace_multifield(df_copy)
    # replace specific fields
    df_copy = replace_specific(df_copy)
    # fill freq nan with median
    df_copy = replace_freq(df_copy, use=freq_fill)
    # transform freq cols
    if transform:
        df_copy = apply_transform(df_copy, use=transform_fn)
    # drop columns with threshold percentage nan
    df_copy.dropna(axis=1, thresh=int(nan_percent*df_copy.shape[0]), inplace=True)
    return df_copy

def replace_noans(df):
    """replace prefer not to say if avaialable and remove object cols"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if col!='label': # exclude label
            # remove time stamp cols
            if df_copy[col].dtype==object:
                df_copy.drop(col, axis=1, inplace=True)
            # replace nan with -818 (prefer not to say)
            elif np.any(df_copy[col]==-818):
                df_copy[col].replace({np.nan: -818.}, inplace=True)
    return df_copy

def replace_multifield(df):
    """replace multiple choice fields"""
    df_copy = df.copy()
    categories_multi = [
        '6160',#Leisure/social activities
        '6145',#Illness, injury, bereavement, stress in last 2 years
    ]
    for cat in categories_multi:
        p_cols = [col for col in df_copy.columns if col[:len(cat)+1]==str(cat)+'-']
        for c in p_cols: # replace with none of the above -7
            df_copy[c].replace(np.nan, -7., inplace=True)
    return df_copy

def replace_specific(df):
    """replace specific categories"""
    df_copy = df.copy()
    categories_zero = [
        '20123',#Single episode of probable major depression
        '20124',#Probable recurrent major depression (moderate)
        '20125', #Probable recurrent major depression (severe)
        '20481', #Self-harmed in past year
        '20484', #Attempted suicide in past year
        '20122', #Bipolar disorder status
        '20126', #Bipolar and major depression status
                 ]
    categories_nts = [
        '20414', #Frequency of drinking alcohol
    ]
    categories_to = [
        '20246', #Trail making completion status
        '20245', #Pairs matching completion status
        '20244', #Symbol digit completion status
    ]
    for c in df_copy.columns:
        for cat in categories_zero:
            if cat in c: 
                df_copy[c].replace(np.nan, 0., inplace=True)
        for cat in categories_nts:
            if cat in c:
                df_copy[c].replace(np.nan, -818., inplace=True) # treat as prefer not to say
        for cat in categories_to:
            if cat in c:
                df_copy[c].replace(np.nan, 1., inplace=True) # treat as abandoned
    return df_copy

def replace_freq(df, use='median'):
    """replace nan in freq with median"""
    df_copy = df.copy()
    for c in df_copy.columns:
        tmp = df_copy[c].value_counts()
        if tmp.shape[0]>7 and c!='label': # most likely frequency/idp
            if use == 'median':
                df_copy[c].fillna(df_copy[c].median(), inplace=True)
            elif use == 'mean':
                df_copy[c].fillna(df_copy[c].mean(), inplace=True)
        elif tmp.shape[0]<=7 and c!='label': # other types of freq
            if np.any(df_copy[c]==-3.) or np.any(df_copy[c]==-1.): # prefer not to say
                df_copy[c].replace({np.nan: -3.}, inplace=True)
#             elif np.any(df_copy[c]==-600.): # degree of bother, also has prefer not to say
#                 df_copy[c].replace({np.nan: -818.}, inplace=True)
    return df_copy

def dummify_qsidp(df):
    """dummify qsidp"""
    # check continuous vs categorical
    all_ls = []
    for i,r in df.iteritems():
        cat_count = len(r.value_counts().values)
        if cat_count < 8:
    #         tmp = pd.get_dummies(r, prefix=i)
            tmp = pd.get_dummies(r, prefix=i, drop_first=True)
        else:
            tmp = pd.DataFrame(r)
        all_ls.append(tmp.reset_index(drop=True))
    df_out = pd.concat(all_ls, axis=1)
    return df_out

def proc_qsidp(df_qsidp, df_featout_ex, questionnaire='all', idp=None):
    """process qsidp"""
    # select 
    qs = load_qscode(questionnaire=questionnaire, idp=idp)
    df_qs = extract_qs(df_qsidp, df_questionnaire=qs, visits=[2])
    # section used ones before imputing
    df_qs_sec = df_qs[df_qs['eid'].isin(df_featout_ex['eid'])]
    # impute qs
    df_qs_imputed = impute_qs(df_qs_sec, nan_percent=0.9, freq_fill='median', 
              transform=False, transform_fn='sqrt')
    # dummify
    df_qs_imputed_dum = dummify_qsidp(df_qs_imputed)
    return df_qs_imputed_dum
    
def cv_loop(bestIC, qs_ls, idp_ls, 
           bfloutput_dir='/well/seymour/users/uhu195/python/pain/output_patients_500',
           curr_dir = '/well/seymour/users/uhu195/python/extract_npy'):
    """cv loop of all possible input combinations"""
    # load bfl
    d = f'Result_IC{bestIC}'
    data_dir = os.path.join(bfloutput_dir, d)
    df_out = full_labels('patients_pain', save=False)
    df_featout_ex = remove_subjs(data_dir, df_out) # remove multiple conditions
    # load qsidp (section to match bfl, impute, dummify)
    if qs_ls is not None or idp_ls is not None:
        df_qsidp = pd.read_csv(os.path.join(curr_dir,'qsidp','qsidp_patients_pain.csv'))
        df_qs_imputed_dum = proc_qsidp(df_qsidp, df_featout_ex, questionnaire=qs_ls, idp=idp_ls)
        print(f'df_qs_imputed_dum shape={df_qs_imputed_dum.shape}')
        # merge bfl and qsidp
        df_bfl_qsidp = df_featout_ex.merge(df_qs_imputed_dum, left_on='eid', right_on='eid', how='left',indicator=False)
    else:
        df_bfl_qsidp = df_featout_ex
    print(f'df_bfl_qsidp shape={df_bfl_qsidp.shape}')

    # retrain params 
    import optuna
    from compare_hyperparams import objective, load_feats

    X_train, y_train = load_feats(df_bfl_qsidp, bestIC, train=False, balance=True, scaler=True)
    print(X_train.shape, y_train.shape)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)
    bp = study.best_trial.params
    print(bp)
    
    # cv results
    df_cv = cv_classify(df_bfl_qsidp, bestIC, classifier='rforest', tuned_params=bp, cv_fold=4, scaler=True, balance=True)
    return df_cv

def combinations_all(a):
    """return all combinations of list"""
    from itertools import combinations
    ls = []
    for i in range(0,len(a)+1):
        ls += list(combinations(a,i))
    return ls

def match_question(q_codes, questionnaire='all', idp='all'):
    """backward search questions to match question code"""
    df_qs = load_qscode(questionnaire=questionnaire, idp=idp)
    question_ls = []
    for c in q_codes:
        if isinstance(c, str):
            code = int(c.split('-')[0])
            question = df_qs[df_qs['code']==code]['Field title'].values
            question_ls.append(question)
    return question_ls

# run
if __name__=="__main__":
    bfl_dir = '/well/seymour/users/uhu195/python/pain/'
    # bfloutput_dir = os.path.join(bfl_dir, 'output_patients_50')
#     bfloutput_dir = os.path.join(bfl_dir, 'output_patients_500')
    bfloutput_dir = os.path.join(bfl_dir, 'output_patients_exmult_500')
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    

    # loop through
    IC = int(sys.argv[1]) # to split up work to different qsub
#     IC_all = [30, 50, 70, 100, 200, 500]

    add_ls = str(sys.argv[2]) # to split up work to different qsub
    if add_ls == 'qs':
        add_all = ['cognitive','demographic','lifestyle','mental']
        added_ls = combinations_all(add_all)
        idp_in = None
    else:
        add_all = ['t1vols','subcorticalvol','fast','t2star','wdmri','taskfmri']#'dmri','t2weighted',
        added_ls = combinations_all(add_all)
        qs_in = None
    
    
    cv_res = []

    for feat in added_ls:
        # clean up empty qs
        if len(list(feat))==0:
            feat_in = None
        else:
            feat_in = list(feat)
            
        if add_ls == 'qs':
            qs_in = feat_in
        else:
            idp_in = feat_in

        print(f'Currently running - bestIC={IC}, qs_ls={qs_in}, idp_ls={idp_in}')
        df_cv = cv_loop(IC, qs_in, idp_in)
        df_cv['bestIC'] = IC
        df_cv['qsidp'] = str(feat_in)
        cv_res.append(df_cv)
                
    df_save = pd.concat(cv_res)
    df_save.to_csv(os.path.join(curr_dir, 'cv_results', 'paintype', f'cv_results_{IC}IC_{add_ls}.csv'), index=None)

