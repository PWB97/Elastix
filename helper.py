import numpy as np
import SimpleITK as sitk
from skimage import transform
import os
import tre3d
import random
from scipy import ndimage
import math

mode = 'E1'

BASE_64_DIR = '/mnt/hd3/new3dDataset_64/'
TRAIN_64 = BASE_64_DIR + 'train/'
TEST_64 = BASE_64_DIR + 'test/'
PATH = '/home/puwenbo/puwenbo/Dataset/new3dDataset/test/'

if mode == 'E1':
    TEST_64 = TEST_64 + 'E1/'
    OUTPUT = '/home/puwenbo/puwenbo/Result/E1/'

if mode == 'E2':
    TEST_64 = TEST_64 + 'E2/'
    OUTPUT = '/home/puwenbo/puwenbo/Result/E2/'


p = sitk.ParameterMap()
p['FixedInternalImagePixelType'] = ['float']
p['MovingInternalImagePixelType'] = ['float']
p['FixedImageDimension'] = ['3']
p['MovingImageDimension'] = ['3']
p['UseDirectionCosines'] = ['true']
p['Registration'] = ['MultiResolutionRegistration']
p['Interpolator'] = [ "BSplineInterpolator"]
p['ResampleInterpolator'] = ["FinalBSplineInterpolator"]
p['Resampler'] = ["DefaultResampler"]
p['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
p['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]
p['Optimizer'] = ["AdaptiveStochasticGradientDescent"]
p['Transform'] =  ["EulerTransform"]
p['Metric'] = ["AdvancedMattesMutualInformation"]
p['AutomaticScalesEstimation'] = [ "true"]
p['AutomaticTransformInitialization'] = ["true"]
p['HowToCombineTransforms'] = ["Compose"]
p['NumberOfHistogramBins']=['32']
p['ErodeMask'] = ['false']
p['NumberOfResolutions'] = ['4']
p['MaximumNumberOfIterations'] = ['250']
p['NumberOfSpatialSamples'] = ['2048']
p['NewSamplesEveryIteration'] = ['true']
p['ImageSampler'] = ['Random']
p['BSplineInterpolationOrder'] = ['1']
p['FinalBSplineInterpolationOrder'] = ['3']
p['DefaultPixelValue'] = ['0']


def get64():
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    for dir in os.listdir('/mnt/hd3/new3dDataset/test'):
        if not os.path.exists(PATH+dir):
            os.mkdir(PATH+dir)
        for CT in os.listdir('/mnt/hd3/new3dDataset/test/'+dir+'/CT'):
            imageCT = sitk.ReadImage('/mnt/hd3/new3dDataset/test/'+dir+'/CT/'+CT)
            imageData = sitk.GetArrayFromImage(imageCT)
            res = transform.resize(imageData, (64,64,64))
            if not os.path.exists(PATH+dir+'/CT/'):
                os.mkdir(PATH+dir+'/CT/')
            sitk.WriteImage(sitk.GetImageFromArray(res), PATH+dir+'/CT/'+CT)
        for MR in os.listdir('/mnt/hd3/new3dDataset/test/'+dir+'/MR'):
            imageMR = sitk.ReadImage('/mnt/hd3/new3dDataset/test/' + dir + '/MR/' + MR)
            imageData = sitk.GetArrayFromImage(imageMR)
            res = transform.resize(imageData, (64, 64, 64))
            if not os.path.exists(PATH + dir + '/MR/'):
                os.mkdir(PATH + dir + '/MR/')
            sitk.WriteImage(sitk.GetImageFromArray(res), PATH + dir + '/MR/' + MR)

def getMovingIMG():
    for patient in os.listdir(TEST_64 + 'ground'):
        os.mkdir(OUTPUT + patient) if not os.path.exists(OUTPUT + patient) else print(patient)
        image = sitk.ReadImage(TEST_64+'ground/'+patient)
        imageData = sitk.GetArrayFromImage(image)
        T = getRandomT()
        # print(T)
        res = ndimage.affine_transform(imageData, tre3d.get_transform_matrix(T))
        sitk.WriteImage(sitk.GetImageFromArray(normalization(res)), OUTPUT + patient + '/moving.nii')
        np.savetxt(OUTPUT + patient + '/T', T)

def getRandomT():
    shift_z = random.randint(-10, 10)
    shift_y = random.randint(-10, 10)
    shift_x = random.randint(-10, 10)
    rotation_z = random.randint(-20, 20)
    # rotation_y = random.randint(-45, 45)
    # rotation_z = random.randint(-45, 45)
    # rotation_x = 0
    rotation_y = 0
    rotation_x = 0
    T = np.array([shift_z, shift_y, shift_x, rotation_z, rotation_y, rotation_x, 1])
    return T

def normalization(new_img):
    # ary = sitk.GetArrayFromImage(img)
    # new_img = np.copy(img)
    new_img[new_img < 0] = 0
    new_img = (new_img - np.min(new_img)) / (np.max(new_img) - np.min(new_img))
    return new_img


def elastix():
    for patient in os.listdir(TEST_64 + '/fixed'):
        try:
            elastixImageFilter = sitk.ElastixImageFilter()
            fix = sitk.ReadImage(TEST_64 + '/fixed/' + patient)
            fix_data = sitk.GetArrayFromImage(fix)
            fix_norm = sitk.GetImageFromArray(normalization(fix_data))
            elastixImageFilter.SetFixedImage(fix_norm)
            elastixImageFilter.SetMovingImage(sitk.ReadImage(OUTPUT + patient + '/moving.nii'))
            elastixImageFilter.SetParameterMap(p)
            if not os.path.isdir(OUTPUT+patient):
                os.mkdir(OUTPUT+patient)
            elastixImageFilter.SetOutputDirectory(OUTPUT+patient)
            elastixImageFilter.Execute()
            res = normalization(sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()))
            sitk.WriteImage(sitk.GetImageFromArray(res), OUTPUT+patient+'/output.nii')
        except Exception:
            print(patient)

def getTrans(patient):
    file = open(OUTPUT + patient + '/TransformParameters.0.txt')
    TransformParameters = []
    for line in file:
        if line.startswith('(TransformParameters'):
            matrix = line.split(')')[0]
            matrix = matrix.split(' ')
            matrix = matrix[-6:]
            for i in range(6):
                if i > 2:
                    TransformParameters.append(math.degrees(float(matrix[5 - i])))
                else:
                    TransformParameters.append(float(matrix[5 - i]))
            TransformParameters.append(1)
            # print(TransformParameters)
            break
    file.close()
    return  TransformParameters


def get_t(patient):
    T = []
    patient = patient.split('.')[0]
    path = TEST_64
    with open(path+'ground2movingparameters.txt', 'r') as file:
        for line in file.readlines():
            parts = line.split('\t')
            name = parts[0]
            if patient == name:
                t = parts[1].strip('\n').strip('[').strip(']')
                t = t.split(' ')
                for i in t:
                    if i != '':
                        T.append(float(i))
                break
    return T

def getTRE(patient):
    TransformParameters = getTrans(patient)
    # T = get_t(patient)
    T = np.loadtxt(OUTPUT + patient + '/T')
    print(T)
    trans = np.array(TransformParameters, dtype=np.float32)
    # print(math.degrees(TransformParameters[3]))
    print(TransformParameters)
    res = tre3d.cal_tre_3d([64, 64, 64], T, trans)
    return res
#
# def getDice(patient):
#