"""
preparing matrices for bigflica
"""
import os, sys
import numpy as np
import pandas as pd
import nibabel as nib


def extract_nii(m_name, sj_csv, n_sj=2, save_dir=None):
    all_m = []
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    data_dir = '/well/win-biobank/projects/imaging/data/data3/subjectsAll'
    
    dff = pd.read_csv(os.path.join(curr_dir, 'bmrc_full', sj_csv))
    # save
    dir_name = sj_csv.split('.')[0]
    if save_dir is None:
        save_dir = os.path.join(curr_dir,'npy',dir_name)
    else:
        save_dir = os.path.join(save_dir,dir_name)
        
    if n_sj is None:
        n_sj = dff.shape[0]
        print(n_sj)
        
    # load mask
    f_path = os.path.join(data_dir, '2'+str(dff['eid'].iloc[0]), m_name)
    img_tmp = nib.load(f_path).get_fdata()
    mask_dat = load_mask(img_tmp)
        
    if m_name != 'fMRI/rfMRI_25.dr/dr_stage2.nii.gz':
        for sj in dff['eid'].iloc[:n_sj]:
            f_path = os.path.join(data_dir, '2'+str(sj), m_name)
            if os.path.exists(f_path):
                img = nib.load(f_path)
                img_dat = img.get_fdata()
#                 print(img_dat.shape)
                img_masked = img_dat[mask_dat]
#                 print(img_masked.shape)
                img_rav = img_masked.reshape(1,-1)
#                 print(img_rav.shape)
                all_m.append(img_rav)
        all_mr = np.vstack(all_m)
        print(all_mr.shape)

        try:
            os.mkdir(save_dir)
        except (FileExistsError):
            pass
        except Exception as e:
            raise e

        save_name = m_name.split('/')[-1].split('.')[0] + '.npy'
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, all_mr.astype('float32'))
    else:
        extract_nii_dr(m_name, sj_csv, n_sj, save_dir)
    
def load_mask(img_data):
    """load mask and return mask"""
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    
    # check image shape
    img_len = img_data.shape[0]
    
    if img_len < 100: 
        mask_path = os.path.join(curr_dir, 'MNI152_T1_2mm_brain.nii.gz')
    else:
        mask_path = os.path.join(curr_dir, 'MNI152_T1_1mm_brain.nii.gz')
    mask_dat = nib.load(mask_path).get_fdata()>0
    return mask_dat

def extract_nii_dr(m_name, sj_csv, n_sj, save_dir=None):
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    data_dir = '/well/win-biobank/projects/imaging/data/data3/subjectsAll'
    
    dff = pd.read_csv(os.path.join(curr_dir, 'bmrc_full', sj_csv))
    # save
    dir_name = sj_csv.split('.')[0]
    if save_dir is None:
        save_dir = os.path.join(curr_dir,'npy',dir_name)
    else:
        save_dir = save_dir
    print(save_dir)
        
    # load mask
    f_path = os.path.join(data_dir, '2'+str(dff['eid'].iloc[0]), m_name)
    img_tmp = nib.load(f_path).get_fdata()
    mask_dat = load_mask(img_tmp)

    if m_name == 'fMRI/rfMRI_25.dr/dr_stage2.nii.gz':
        for i in range(25):
            all_m = []
            for sj in dff['eid'].iloc[:n_sj]:
                f_path = os.path.join(data_dir, '2'+str(sj), m_name)
                if os.path.exists(f_path):
                    img = nib.load(f_path)
                    img_dat = img.get_fdata()[:,:,:,i]
#                     print(img_dat.shape)
                    img_masked = img_dat[mask_dat]
#                     print(img_masked.shape)
                    img_rav = img_masked.reshape(1,-1)
#                     print(img_rav.shape)
#                     img_rav = img_dat[:,:,:,i].reshape(1,-1)
#                     print(img_rav.shape)
                all_m.append(img_rav)
            all_mr = np.vstack(all_m)
#             print(all_mr.shape)

            save_name = m_name.split('/')[-1].split('.')[0] + f'_{i}.npy'
            save_path = os.path.join(save_dir, save_name)
            np.save(save_path, all_mr.astype('float32'))
    
# run
if __name__=="__main__":
    # load from
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    data_dir = '/well/win-biobank/projects/imaging/data/data3/subjectsAll'

    # arg
    mod_num = int(sys.argv[1])
#     sj_num = int(sys.argv[2])
    sj_num = None
    
    # modalities
    mods = [
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
        'T2_FLAIR/lesions/final_mask_to_MNI.nii.gz',
        'fMRI/rfMRI_25.dr/dr_stage2.nii.gz'
    ]
    
    ## sj_csv = 'subjs_patients_pain_restricted.csv'
#     sj_csv = 'subjs_patients_pain_restricted_matched.csv'
#     sj_csv = 'subjs_patients_pain.csv'
    ##sj_csv = 'subjs_patients_pain_matched.csv'
#     sj_csv = 'subjs_digestive.csv'
#     sj_csv = 'subjs_patients_pain_exmult.csv'
    sj_csv = 'subjs_paincontrol.csv'

    shared_dir = '/well/tracey/shared/fps-ukb/'
    extract_nii(mods[mod_num], sj_csv, n_sj=sj_num, save_dir=shared_dir)
