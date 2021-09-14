"""
preparing matrices for bigflica
"""
import os, sys
import numpy as np
import pandas as pd
import nibabel as nib


def extract_nii(m_name, sj_csv, n_sj=2):
    all_m = []
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'

    dff = pd.read_csv(os.path.join(curr_dir, 'bmrc_full', sj_csv))

    if n_sj is None:
        n_sj = dff.shape[0]
        
    if m_name != 'fMRI/rfMRI_25.dr/dr_stage2.nii.gz':
        for sj in dff['eid'].iloc[:n_sj]:
            f_path = os.path.join(data_dir, '2'+str(sj), m_name)
            if os.path.exists(f_path):
                img = nib.load(f_path)
                img_dat = img.get_fdata()
                print(img_dat.shape)
                img_rav = img_dat.reshape(1,-1)
    #             print(img_rav.shape)
                all_m.append(img_rav)
        all_mr = np.vstack(all_m)
        print(all_mr.shape)
        # save
        dir_name = sj_csv.split('.')[0]
        save_dir = os.path.join(curr_dir,'npy',dir_name)
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
        extract_nii_dr(m_name, sj_csv, n_sj)
    
def extract_nii_dr(m_name, sj_csv, n_sj):
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'

    dff = pd.read_csv(os.path.join(curr_dir, 'bmrc_full', sj_csv))

    if m_name == 'fMRI/rfMRI_25.dr/dr_stage2.nii.gz':
        for i in range(25):
            all_m = []
            for sj in dff['eid'].iloc[:n_sj]:
                f_path = os.path.join(data_dir, '2'+str(sj), m_name)
                if os.path.exists(f_path):
                    img = nib.load(f_path)
                    img_dat = img.get_fdata()
#                     print(img_dat.shape)
                    img_rav = img_dat[:,:,:,i].reshape(1,-1)
#                     print(img_rav.shape)
                all_m.append(img_rav)
            all_mr = np.vstack(all_m)
#             print(all_mr.shape)
            # save
            dir_name = sj_csv.split('.')[0]
            save_dir = os.path.join(curr_dir,'npy',dir_name)

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
    sj_num = int(sys.argv[2])
    
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
    
    sj_csv = 'subjs_patients_pain_restricted.csv'
    extract_nii(mods[mod_num], sj_csv, n_sj=sj_num)
