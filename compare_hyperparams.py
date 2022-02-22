"""
compare hyperparameters of bfl
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from nilearn.plotting import plot_stat_map#,plot_anat,plot_epi

def full_labels(csv_name, save=True):
    """getting full set of labels for features"""
    # load data
    base_dir = '/well/seymour/users/uhu195/python/extract_npy/'
    print(f'{base_dir}/labels/label_{csv_name}.csv')
    labels = pd.read_csv(f'{base_dir}/labels/label_{csv_name}.csv')
    sj_bmrc = pd.read_csv(f'{base_dir}/bmrc_subjs/subjs_{csv_name}.csv', header=None)
    sj_full = pd.read_csv(f'{base_dir}/bmrc_full/subjs_{csv_name}.csv')
    # rename columns
    sj_bmrc.rename(columns={0:'bmrc'}, inplace=True)
    sj_full.rename(columns={'eid':'bmrc_eid'}, inplace=True)
    print(sj_full.shape)
    # merge
    bmrc_full = sj_bmrc.merge(sj_full, left_on='bmrc', right_on='bmrc_eid', how='left',indicator=True)
    # concat with labels
    full_df = pd.concat([labels, bmrc_full], axis=1)
    df_out = full_df[full_df['_merge']=='both']
    df_out_clean = df_out.drop(columns=['bmrc_eid', '_merge']).reset_index(drop=True)
    print(df_out_clean.shape)
    # save
    if save:
        save_path = f'{base_dir}/labels_full/label_{csv_name}.csv'
        df_out_clean.to_csv(save_path, index=None)
    return df_out_clean

def remove_subjs(feat_path, df_label, remove_dup=True):
    """remove participants with multiple conditions"""
    feats = np.load(os.path.join(feat_path, 'subj_course.npy'))
    if remove_dup:
        exclude = df_label[df_label[['irritable bowel syndrome', 'back pain', 'migraine','osteoarthritis']].sum(axis=1)>1]['eid']
        df_featout = pd.concat([df_label, pd.DataFrame(feats)], axis=1)
        df_featout_ex = df_featout[~df_featout['eid'].isin(exclude.values)]
    else:
        df_featout_ex = pd.concat([df_label, pd.DataFrame(feats)], axis=1)
    return df_featout_ex

def data_prep(df):
    """prepare df to x and y for clf"""
    # dummify labels
    dfc = df.copy()
    rm_cols = ['irritable bowel syndrome', 'migraine', 'back pain', 'osteoarthritis', 'label', 'eid', 'bmrc']
    rm_idx = [item in df.columns for item in rm_cols]
    rm_colnames = [item for idx, item in enumerate(rm_cols) if rm_idx[idx]]
    
    # remove redundant cols in X
    if len(rm_colnames)>0:
        X = dfc.drop(columns=rm_colnames).to_numpy()
    else:
        X = dfc.to_numpy()
    print(X.shape)

    # take labels out
    if 'label' in df.columns:
        y = df['label']
    else:
        y_original = df[['irritable bowel syndrome', 'migraine', 'back pain', 'osteoarthritis']]
        y = y_original.idxmax(axis=1)
    return X, y

def cv_classify(df, classifier='dtree', tuned_params=None, cv_fold=10, scaler=True, balance=True):
    """n-fold cross validation classification"""
    from sklearn.model_selection import cross_validate

    X, y = data_prep(df)
    # balance dataset
    if balance:
        from imblearn.under_sampling import RandomUnderSampler
        # define undersampling strategy
        under = RandomUnderSampler(random_state=0)
        # fit and apply the transform
        X, y = under.fit_resample(X, y)
    # apply scaler
    if scaler:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    # define classifier
    if classifier == 'dtree':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=5)
    elif classifier == 'rforest':
        from sklearn.ensemble import RandomForestClassifier
        if tuned_params is not None:
            clf = RandomForestClassifier(**tuned_params)
        else:
            clf = RandomForestClassifier(max_depth=5)
    elif classifier == 'lgb':
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_jobs=-1)
    # cv result
    print(len(np.unique(y)))
    if len(np.unique(y)) <= 2: # binary
        cv_results = cross_validate(clf, X, y, cv=cv_fold, return_train_score=False, scoring=('accuracy', 'f1', 'roc_auc'))
        df_res = pd.DataFrame(cv_results)
        # print res
        print(f"{cv_fold}-fold CV classification with classifier {clf}:\n"
            f"test ROC AUC={df_res['test_roc_auc'].mean():.4f}, test accuracy={df_res['test_accuracy'].mean():.4f}, test f1={df_res['test_f1'].mean():.4f}")
    else:
        cv_results = cross_validate(clf, X, y, cv=cv_fold, return_train_score=False, scoring=('accuracy', 'f1_micro', 'roc_auc_ovo'))
        df_res = pd.DataFrame(cv_results)
        # print res
        print(f"{cv_fold}-fold CV classification with classifier {clf}:\n"
            f"test ROC AUC={df_res['test_roc_auc_ovo'].mean():.4f}, test accuracy={df_res['test_accuracy'].mean():.4f}, test f1={df_res['test_f1_micro'].mean():.4f}")

    return df_res

def load_mask(img_data):
    """load mask and return mask"""
    import nibabel as nib
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    
    # check image shape
    img_len = img_data.shape[0]
    
    if img_len < 100: 
        mask_path = os.path.join(curr_dir, 'MNI152_T1_2mm_brain.nii.gz')
    else:
        mask_path = os.path.join(curr_dir, 'MNI152_T1_1mm_brain.nii.gz')
    mask_dat = nib.load(mask_path).get_fdata()>0
    return mask_dat

def feature_importance(clf, feature_names, plot=True):
    """plot feature importance and return series"""
    import time
    import numpy as np

    start_time = time.time()
    importances = clf.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in clf.estimators_], axis=0)
    # importance
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(importances, index=feature_names)
    # plot
    if plot:
        fig, ax = plt.subplots()
        feat_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
    return feat_importances.sort_values(ascending=False)

def top_mods(data_dir, feature_importance, mod_num=3, plot=True):
    """return top modalities given feature importance"""
    mod_contrib = np.load(os.path.join(data_dir, 'mod_contribution.npy')) # mod x feats
    top_feat = int(feature_importance.index[0].split(' ')[1])
    print(f'top feature: {top_feat}')
    topIC = mod_contrib[:,top_feat]
    # load mod names
    mod_names = pd.read_csv('./sorted_feats.csv')
    plot_mod = mod_names.iloc[topIC.argsort()]
#     print(plot_mod.index.max())
    print('top modalities', plot_mod.iloc[-mod_num:])
    if plot:
        import matplotlib.pyplot as plt
        plt.subplots(figsize=(8,4))
        plt.bar(np.arange(len(topIC)), topIC[topIC.argsort()])
        plot_mod_names = plot_mod['modalities'].values
        plt.xticks(np.arange(len(plot_mod_names)), plot_mod_names, rotation=90)
    return top_feat, plot_mod.index[-mod_num:].values[::-1]


def compare_patients(output_dir, save=True):
    """compare patients and bfl hyperparams nlats"""
    df_out = full_labels('patients_pain', save=False)
    all_res = []
    for d in os.listdir(output_dir):
        if d.startswith('R'):
            data_dir = os.path.join(output_dir, d)
            print(data_dir)
            df_featout_ex = remove_subjs(data_dir, df_out) # remove multiple conditions
            print(df_featout_ex.shape)
            IC_num = int(d.split('_')[-1][2:])
            df_res = cv_classify(df_featout_ex, classifier='rforest', cv_fold=10, scaler=True, balance=True)
            df_res['IC number'] = IC_num
            all_res.append(df_res)
    df_all_res = pd.concat(all_res)
    if save:
        curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
        df_all_res.to_csv(os.path.join(curr_dir,'hyperparam_cv', 'patients.csv'))
    return df_all_res

def objective(trial, X, y):
    """tuning using optuna"""
    from sklearn.model_selection import train_test_split
    import sklearn.ensemble
    import sklearn.model_selection
    
#     X, y = load_feats(dataset='patient', train=False)
#     X, y = load_feats(data_input, IC_num, train=False)

#     classifier_name = trial.suggest_categorical("classifier", ["RandomForest"])
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

    score = sklearn.model_selection.cross_val_score(classifier_obj, X, y, 
                                                    n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

def load_feats(data, test_size=0.5, dummies=False, 
               train=True, balance=False, scaler=False):
    """load dataset from csv, return full or train sets"""
    # prep by renaming columns and split xy
    X, y = data_prep(data)
    
    # balance dataset
    if balance:
        from imblearn.under_sampling import RandomUnderSampler
        # define undersampling strategy
        under = RandomUnderSampler(random_state=0)
        # fit and apply the transform
        X, y = under.fit_resample(X, y)
    # apply scaler
    if scaler:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    # dummify 
    if dummies:
        y = pd.get_dummies(y)
        
    # return full or train only
    if train:
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)
        return X_train, y_train, X_test, y_test
    else:
        return X, y
    
def compare_patients_optuna(output_dir, save=True):
    """compare patients and bfl hyperparams nlats"""
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'

    df_out = full_labels('patients_pain', save=False)
    all_res = []
    for d in os.listdir(output_dir):
        if d.startswith('R'):
            data_dir = os.path.join(output_dir, d)
            print(data_dir)
            df_featout_ex = remove_subjs(data_dir, df_out) # remove multiple conditions
            print(df_featout_ex.shape)
            IC_num = int(d.split('_')[-1][2:])
            # using half train set to tune
#             X_train, y_train, X_test, y_test = load_feats(df_featout_ex, IC_num, train=True, balance=True, scaler=True)
            # using whole dataset to tune
            X_train, y_train = load_feats(df_featout_ex, IC_num, train=False, balance=True, scaler=True)
            print(X_train.shape, y_train.shape)
            
            # using optuna
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)
            bp = study.best_trial.params
            print(bp)

            df_res = cv_classify(df_featout_ex, IC_num=IC_num, 
                                 classifier='rforest', tuned_params=bp, 
                                 cv_fold=10, scaler=True, balance=True)
            df_res['IC number'] = IC_num
            all_res.append(df_res)
            # save bp
            bp_save = os.path.join(curr_dir, 'hyperparam_cv', f'best_params_IC{IC_num}.npy')
            np.save(bp_save, bp)

    df_all_res = pd.concat(all_res)
    if save:
        df_all_res.to_csv(os.path.join(curr_dir,'hyperparam_cv', 'patients_optuna.csv'))
    return df_all_res
    
def feature_importance(clf, feature_names, 
                       plot=True, save_plot=True, 
                       curr_dir='/well/seymour/users/uhu195/python/extract_npy'):
    """plot feature importance and return series"""
    import time
    import numpy as np

    start_time = time.time()
    importances = clf.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in clf.estimators_], axis=0)
    # importance
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(importances, index=feature_names)
    # plot
    if plot:
        fig, ax = plt.subplots()
        feat_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(curr_dir, 'figs', f'feature_importance_IC{len(feature_names)}.png'), bbox_inches='tight')
    return feat_importances.sort_values(ascending=False)

def top_mods(data_dir, feature_importance, mod_num=3, plot=True, save_plot=True,
            curr_dir='/well/seymour/users/uhu195/python/extract_npy'):
    """return top modalities given feature importance"""
    mod_contrib = np.load(os.path.join(data_dir, 'mod_contribution.npy')) # mod x feats
    top_feat = int(feature_importance.index[0].split(' ')[1])
    print(f'top feature: {top_feat}')
    topIC = mod_contrib[:,top_feat]
    # load mod names
    mod_names = pd.read_csv(os.path.join(curr_dir, 'sorted_feats.csv'))
    plot_mod = mod_names.iloc[topIC.argsort()]
#     print(plot_mod.index.max())
    print('top modalities', plot_mod.iloc[-mod_num:])
    if plot:
        import matplotlib.pyplot as plt
        plt.subplots(figsize=(8,4))
        plt.bar(np.arange(len(topIC)), topIC[topIC.argsort()])
        plot_mod_names = plot_mod['modalities'].values
        plt.xticks(np.arange(len(plot_mod_names)), plot_mod_names, rotation=90)
        if save_plot:
            plt.savefig(os.path.join(curr_dir, 'figs', f'topfeature_{top_feat}_modsrank.png'), bbox_inches='tight')
    return top_feat, plot_mod.index[-mod_num:].values[::-1]

def load_modZ(data_dir, modality_num, feature_num, plot_threshold, 
              plot_coords=[0, 0, 0], plot=True, save_plot=True,
              curr_dir='/well/seymour/users/uhu195/python/extract_npy'):
    """load modality Z map given number"""
    import nibabel as nib
    from nibabel import Nifti1Image

    df_mod = np.load(os.path.join(data_dir, f'flica_mod{modality_num+1}_Z.npy'))
    print(df_mod.shape)
    img_data = df_mod[:,feature_num]
    # load mask by shape
    if df_mod.shape[0]>1e6:
        mask_path = os.path.join(curr_dir, 'MNI152_T1_1mm_brain.nii.gz')
    else:
        mask_path = os.path.join(curr_dir, 'MNI152_T1_2mm_brain.nii.gz')
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    mask = np.where(mask_data>0)
    # reshape z map
    img_reshape = np.zeros(mask_data.shape)
    img_reshape[mask] = img_data
    # reconstruct using MNI affine
    img_reshape_ni = Nifti1Image(img_reshape, affine=mask_img.affine)
    # load modality names
    mod_names = pd.read_csv(os.path.join(curr_dir, 'sorted_feats.csv'))
    # plotting
    if plot:
        plot_mod = mod_names.iloc[modality_num].values
        plot_stat_map(img_reshape_ni, bg_img=mask_path, threshold=plot_threshold, cut_coords=plot_coords, title=plot_mod[0])
        if save_plot:
            plt.savefig(os.path.join(curr_dir, 'figs', f'mod_{plot_mod[0]}.png'), bbox_inches='tight')
    return img_reshape_ni, mask_path

def best_clf(param_csv, bfloutput_dir, curr_dir='/well/seymour/users/uhu195/python/extract_npy'):
    """build best clf from optuna params"""
    from sklearn.ensemble import RandomForestClassifier

    df_cv = pd.read_csv(param_csv)
    df_comp = df_cv.groupby(['IC number'])[['test_roc_auc_ovo', 'test_accuracy']].mean()
    bestIC = df_comp.index[df_comp['test_roc_auc_ovo'].argmax()]
    print(f'Best performing number of IC = {bestIC}')
    bp_path = os.path.join(curr_dir, 'hyperparam_cv', f'best_params_IC{bestIC}.npy')
    params = np.load(bp_path, allow_pickle='TRUE').item() # load dict
    # load data
    d = f'Result_IC{bestIC}'
    data_dir = os.path.join(bfloutput_dir, d)
    df_out = full_labels('patients_pain', save=False)
    df_featout_ex = remove_subjs(data_dir, df_out) # remove multiple conditions
#     X_train, y_train, _, _ = load_feats(df_featout_ex, bestIC, train=True, balance=True, scaler=True)
    X_train, y_train = load_feats(df_featout_ex, bestIC, train=False, balance=True, scaler=True)
    # train clf
    forest = RandomForestClassifier(**params)
    # load data
    forest.fit(X_train, y_train)
    # plot feature importance
    feature_names = [f'feature {i}' for i in range(bestIC)]
    feature_imp = feature_importance(forest, feature_names, plot=True, save_plot=True)
    # plot top mods given most important feature
    top_feat, mod_ls = top_mods(data_dir, feature_imp, mod_num=5)
    # plot modalities 
    for mod in mod_ls:
        print(f'modality {mod}')
        reshape_z, mask_path = load_modZ(data_dir, modality_num=mod, feature_num=top_feat, plot_threshold=40)

def load_cv(cv_dir, qtype='all'):
    """load cv results"""
    cv_ls = []

    if qtype=='idp':
        end_name = '_idp.csv'
    elif qtype=='qs':
        end_name = '_qs.csv'
    elif qtype=='qsidp':
        end_name = '_qsidp.csv'
    else:
        end_name = '.csv'
        
    for f in os.listdir(cv_dir):
        if f.endswith(end_name):	
            tmp = pd.read_csv(os.path.join(cv_dir, f))
            cv_ls.append(tmp)
    df = pd.concat(cv_ls)
    print(df.head())
    return df


# run
if __name__=="__main__":
    # run optuna to fit best rf for each IC
    bfl_dir = '/well/seymour/users/uhu195/python/pain/'
    bfloutput_dir = os.path.join(bfl_dir, 'output_patients_50')
#     compare_patients(output_dir) # not using optuna
#     compare_patients_optuna(bfloutput_dir) # tune using optuna

    # load best hyperparams and plot feature importance
#     curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
#     param_dir = os.path.join(curr_dir, 'hyperparam_cv', 'patients_optuna.csv')
#     best_clf(param_dir, bfloutput_dir)

    # create list of excluded multiple subjects
    df_out = full_labels('patients_pain', save=False)
    df_featout_ex = remove_subjs(data_dir, df_out) # remove multiple conditions
    