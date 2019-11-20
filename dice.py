from helper import *

for label in os.listdir(TEST_64 + 'label'):
    print('---------------------------')
    print(label)
    l_image = sitk.ReadImage(TEST_64 + 'label/' + label)
    l_data = sitk.GetArrayFromImage(l_image)
    trans = getTrans(label)
    T = np.loadtxt(OUTPUT + label + '/T')
    print(trans)
    print(T)
    trans_1 = ndimage.affine_transform(l_data, tre3d.get_transform_matrix(T))
    # T[0:6] = -T[0:6]
    trans_2 = ndimage.affine_transform(trans_1, tre3d.get_transform_matrix(trans))
    res = sitk.GetImageFromArray(normalization(trans_2))
    sitk.WriteImage(res, OUTPUT + label + '/label_trans.nii')



