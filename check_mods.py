"""
check if subject has all modalities
"""
import os
import numpy as np
import pandas as pd

def check_mods(sj_csv):
    # check modalities
    full_sj = []
    sj_path = './bmrc_subjs/'+sj_csv
    df_sj = pd.read_csv(sj_path, header=None)
    for sj in df_sj[0].values:
        count = 0
        for m in mods:
            f_path = os.path.join(data_dir, '2'+str(sj), m)
            if os.path.exists(f_path):
                count += 1
        if count == len(mods):
            full_sj.append(sj)
#             print(f'subject {sj} has {count} modalities.')
    df_full = pd.DataFrame({'eid':full_sj})
    print(df_full.shape)
    df_full.to_csv('./bmrc_full/'+sj_csv,index=None)
    print(sj_csv, len(full_sj))

# run
if __name__=="__main__":
    # load from
    data_dir = '/well/win-biobank/projects/imaging/data/data3/subjectsAll'
    
    # modalities
    mods = [
    'fMRI/rfMRI_25.dr/dr_stage2.nii.gz',
    'fMRI/tfMRI.feat/reg_standard/stats/zstat1.nii.gz',
    'fMRI/tfMRI.feat/reg_standard/stats/zstat2.nii.gz',
    'fMRI/tfMRI.feat/reg_standard/stats/zstat5.nii.gz',
    'fMRI/tfMRI.feat/reg_standard/stats/cope1.nii.gz',
    'fMRI/tfMRI.feat/reg_standard/stats/cope2.nii.gz',
    'fMRI/tfMRI.feat/reg_standard/stats/cope5.nii.gz',
    'dMRI/TBSS/stats/all_FA_skeletonised.nii.gz',
    'dMRI/TBSS/stats/all_ICVF_skeletonised.nii.gz',
    'dMRI/TBSS/stats/all_ISOVF_skeletonised.nii.gz',
    'dMRI/TBSS/stats/all_L1_skeletonised.nii.gz',
    'dMRI/TBSS/stats/all_L2_skeletonised.nii.gz',
    'dMRI/TBSS/stats/all_L3_skeletonised.nii.gz',
    'dMRI/TBSS/stats/all_MD_skeletonised.nii.gz',
    'dMRI/TBSS/stats/all_MO_skeletonised.nii.gz',
    'dMRI/TBSS/stats/all_OD_skeletonised.nii.gz',
    'dMRI/autoptx_preproc/tractsNormSummed.nii.gz',
    'T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz',
    'T1/transforms/T1_to_MNI_warp_jac.nii.gz',
    'SWI/T2star_to_MNI.nii.gz',
    'T2_FLAIR/lesions/final_mask_to_MNI.nii.gz'
    ]
    
    # check
    for f in os.listdir('./bmrc_subjs'):
        if f.endswith('.csv'):
            print(f)
            check_mods(f)