"""
combine npys from patient and control
"""

import os
import numpy as np

# load from
curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
npy_dir = os.path.join(curr_dir, 'npy')

# patient vs match
patient_gp = 'subjs_patients_pain_restricted'
patient_dir = os.path.join(npy_dir, patient_gp)
matched_dir = os.path.join(npy_dir, patient_gp+'_matched')

# save
dir_name = patient_gp.split('.')[0] + '_combined'
save_dir = os.path.join(curr_dir,'npy',dir_name)
try:
    os.mkdir(save_dir)
except (FileExistsError):
    pass
except Exception as e:
    raise e

# list
for f in os.listdir(patient_dir):
    fname = f
    patient_npy = os.path.join(patient_dir, f)
    pmat = np.load(patient_npy)
    print(pmat.shape)
    matched_npy = os.path.join(matched_dir, f)
    mmat = np.load(matched_npy)
    print(mmat.shape)
    cmat = np.concatenate((pmat, mmat), axis=0)
    print(cmat.shape)
    # saving
    save_name = fname.split('.')[0] + '.npy'
    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, cmat.astype('float32'))