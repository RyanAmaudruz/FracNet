import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm

from dataset.fracnet_dataset import FracNetInferenceDataset
from dataset import transforms as tsfm
from model.unet import UNet
import SimpleITK as sitk



def _get_max_area(imbin):
    """
    Implementation by Kaiming Kuang <kaiming.kuang@dianei-ai.com>	14 Dec 2020 at 04:33

    :param imbin:
    :return:
    """
    labs = label(imbin)
    rpps = sorted(regionprops(labs), key=lambda p: p.area)
    mask = labs == rpps[-1].label

    return mask

def thorax_mask_pipeline(x):
    """
    Implementation by Kaiming Kuang <kaiming.kuang@dianei-ai.com>	14 Dec 2020 at 04:33

    :param imbin:
    :return:
    """
    x = _get_max_area(ndimage.binary_fill_holes(x))

def _get_thorax_mask(image):
    """
    Implementation by Kaiming Kuang <kaiming.kuang@dianei-ai.com>	14 Dec 2020 at 04:33

    :param imbin:
    :return:
    """
    mask = image > -200
    mask = [ndimage.binary_fill_holes(x) for x in mask]
    mask = [_get_max_area(x) for x in mask]
    mask = [mask[i] * (image[i] < -400) for i in range(len(mask))]
    mask = np.stack([ndimage.binary_fill_holes(x) for x in mask])
    return mask


def _rescale(arr, target_shape, interpolation=0):
    """
    Implementation by Kaiming Kuang <kaiming.kuang@dianei-ai.com>	14 Dec 2020 at 04:33

    :param imbin:
    :return:
    """
    target_shape = target_shape[::-1]
    arr = sitk.GetImageFromArray(arr.astype(np.uint8))
    old_spacing = arr.GetSpacing()
    old_shape = arr.GetSize()
    target_spacing = tuple([old_spacing[i] * old_shape[i] / target_shape[i]
                            for i in range(len(target_shape))])

    resample = sitk.ResampleImageFilter()
    interpolator = sitk.sitkLinear if interpolation == 1 \
        else sitk.sitkNearestNeighbor
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(target_shape)
    new_arr = resample.Execute(arr)

    return sitk.GetArrayFromImage(new_arr).astype(np.bool)


def _get_lung_mask(image, shrink_ratio):
    """
    Implementation by Kaiming Kuang <kaiming.kuang@dianei-ai.com>	14 Dec 2020 at 04:33

    :param imbin:
    :return:
    """
    mask = _get_thorax_mask(image)
    old_shape = image.shape
    target_shape = tuple([round(dim * shrink_ratio) for dim in old_shape])
    mask = _rescale(mask, target_shape)

    labs = label(mask)
    rpps = sorted(regionprops(labs), key=lambda p: p.area)
    mask = labs == rpps[-1].label

    if rpps[-2].area > rpps[-1].area / 2:
        mask = mask | (labs == rpps[-2].label)

    xpix = mask.sum((0, 1))
    labs = label(xpix < xpix.mean())
    xcrg = np.where(labs == labs[len(labs) // 2])[0]

    tube = []
    for chil in mask:
        chil = ndimage.binary_erosion(chil, disk(3))
        labs = label(chil)
        rpps = regionprops(labs)
        for p in rpps:
            x = int(p.centroid[-1])
            labs[labs == p.label] = 0 if x not in xcrg else p.label
        tube.append(ndimage.binary_dilation(labs > 0, disk(3)))
    tube = np.stack(tube)

    mask = mask * (tube == 0)
    mask = np.stack([ndimage.binary_closing(x, disk(10)) for x in mask])
    mask = _get_max_area(mask)

    return mask


def _get_lung_contour(image, shrink_ratio):
    """
    Implementation by Kaiming Kuang <kaiming.kuang@dianei-ai.com>	14 Dec 2020 at 04:33

    :param imbin:
    :return:
    """
    old_shape = image.shape
    lung_mask = _get_lung_mask(image, shrink_ratio)
    lung_contour = np.logical_xor(ndimage.maximum_filter(lung_mask, 10),
                                  lung_mask)
    lung_contour = _rescale(lung_contour, old_shape)
    return lung_contour


def _remove_non_rib_pred(pred, image, shrink_ratio):
    """
    Implementation by Kaiming Kuang <kaiming.kuang@dianei-ai.com>	14 Dec 2020 at 04:33

    :param imbin:
    :return:
    """
    # transpose the image and prediction from xyz to zyx
    pred = pred.transpose(2, 1, 0)
    image = image.transpose(2, 1, 0)


    lung_contour = _get_lung_contour(image, shrink_ratio)

    pred = np.where(lung_contour, pred, 0)

    # transpose them back
    pred = pred.transpose(2, 1, 0)
    image = image.transpose(2, 1, 0)

    return pred




def _remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, pred, 0)

    return pred


def _remove_spine_fp(pred, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pred)


def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def _post_process(pred, image, prob_thresh, bone_thresh, size_thresh):

    # remove non-rib predictions
    pred = _remove_non_rib_pred(pred, image, 0.25)

    # remove connected regions with low confidence
    pred = _remove_low_probs(pred, prob_thresh)

    # remove spine false positives
    pred = _remove_spine_fp(pred, image, bone_thresh)

    # remove small connected regions
    pred = _remove_small_objects(pred, size_thresh)

    return pred


def _predict_single_image(model, dataloader, postprocess, prob_thresh,
        bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)
    crop_size = dataloader.dataset.crop_size
    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda()
            output = model(images).sigmoid().cpu().numpy().squeeze(axis=1)

            for i in range(len(centers)):
                center_x, center_y, center_z = centers[i]
                cur_pred_patch = pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ]
                pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ] = np.where(cur_pred_patch > 0, np.mean((output[i],
                    cur_pred_patch), axis=0), output[i])

    if postprocess:
        pred = _post_process(pred, dataloader.dataset.image, prob_thresh,
            bone_thresh, size_thresh)

    return pred


def _make_submission_files(pred, image_id, affine):
    pred_label = label(pred > 0).astype(np.int16)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    # placeholder for label class since classifaction isn't included
    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


def predict(args):
    batch_size = 16
    num_workers = 4
    postprocess = True if args.postprocess == "True" else False

    model = UNet(1, 1, first_out_channels=16)
    model.eval()
    if args.model_path is not None:
        model_weights = torch.load(args.model_path)
        model.load_state_dict(model_weights['model'])
    model = nn.DataParallel(model).cuda()

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]

    image_path_list = sorted([os.path.join(args.image_dir, file)
        for file in os.listdir(args.image_dir) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0]
        for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = FracNetInferenceDataset(image_path, transforms=transforms)
        dataloader = FracNetInferenceDataset.get_dataloader(dataset,
            batch_size, num_workers)
        pred_arr = _predict_single_image(model, dataloader, postprocess,
            args.prob_thresh, args.bone_thresh, args.size_thresh)
        pred_image, pred_info = _make_submission_files(pred_arr, image_id,
            dataset.image_affine)
        pred_info_list.append(pred_info)
        pred_path = os.path.join(args.pred_dir, f"{image_id}-label.nii.gz")
        nib.save(pred_image, pred_path)

        progress.update()

    pred_info = pd.concat(pred_info_list, ignore_index=True)
    pred_info.to_csv(os.path.join(args.pred_dir, "ribfrac-test-pred.csv"),
        index=False)

class FakeArgs:
    def __init__(self):
        self.image_dir = '/home/ryan/PycharmProjects/FracNet/data/test/ribfrac-test-images/'
        self.pred_dir = '/home/ryan/PycharmProjects/FracNet/data/test/pred/batch_16_w_post/'
        self.model_path = '/home/ryan/Downloads/bestmodel.pth'
        self.prob_thresh = 0.1
        self.bone_thresh = 300
        self.size_thresh = 100
        self.postprocess = 'True'



if __name__ == "__main__":
    # import argparse
    #
    #
    prob_thresh = 0.1
    bone_thresh = 300
    size_thresh = 100
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_dir", required=True,
    #     help="The image nii directory.")
    # parser.add_argument("--pred_dir", required=True,
    #     help="The directory for saving predictions.")
    # parser.add_argument("--model_path", default=None,
    #     help="The PyTorch model weight path.")
    # parser.add_argument("--prob_thresh", default=0.1,
    #     help="Prediction probability threshold.")
    # parser.add_argument("--bone_thresh", default=300,
    #     help="Bone binarization threshold.")
    # parser.add_argument("--size_thresh", default=100,
    #     help="Prediction size threshold.")
    # parser.add_argument("--postprocess", default="True",
    #     help="Whether to execute post-processing.")
    # args = parser.parse_args()

    args = FakeArgs()


    predict(args)

