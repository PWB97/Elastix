import SimpleITK as sitk
import tre3d as t
import cv2
from scipy import ndimage
import os,shutil
import helper

def test():
    print('start')
    for patient in os.listdir('/mnt/hd1/puwenbo/Dataset/T1T2/moving/'):
        print('do ' + patient + 'now')
        elastixImageFilter = sitk.ElastixImageFilter()
        fix = sitk.ReadImage('/mnt/hd1/puwenbo/Dataset/T1T2/fixed/' + patient)
        elastixImageFilter.SetFixedImage(fix)
        elastixImageFilter.SetMovingImage(sitk.ReadImage('/mnt/hd1/puwenbo/Dataset/T1T2/moving/' + patient))
        elastixImageFilter.SetParameterMap(helper.p)
        elastixImageFilter.SetOutputDirectory('/mnt/hd1/puwenbo/Dataset/T1T2/output/')
        elastixImageFilter.Execute()
        res = helper.normalization(sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()))
        sitk.WriteImage(sitk.GetImageFromArray(res), '/mnt/hd1/puwenbo/Dataset/T1T2/output/'+ patient)


def move():
    for patient in os.listdir('/mnt/hd1/puwenbo/Dataset/T1T2/fixed/'):
        if patient in os.listdir('/mnt/hd1/leixiaotang/PD1/'):
            shutil.copy('/mnt/hd1/puwenbo/Dataset/T1T2/fixed/' + patient, '/mnt/hd1/puwenbo/Dataset/T1PD/fixed/')


move()

# test()

def size():

    for patient in os.listdir('/mnt/hd1/puwenbo/Dataset/IXI/PD'):
        fix = sitk.ReadImage('/mnt/hd1/puwenbo/Dataset/IXI/PD/' + patient)
        # moving = sitk.ReadImage('/mnt/hd1/puwenbo/Dataset/T1T2/moving/' + patient + '/result.0.nii')
        # print(patient)
        print('fix shape:', sitk.GetArrayFromImage(fix).shape)
        # print('moving shape:', sitk.GetArrayFromImage(moving).shape)
        # print('----------------')


# size()
