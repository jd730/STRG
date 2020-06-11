from torchvision import get_image_backend

from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from datasets.activitynet import ActivityNet
from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5
import pdb

def image_name_formatter(x):
    return 'image_{:05d}.jpg'.format(x)

def sthv2_image_name_formatter(x):
    return '{:06d}.jpg'.format(x)

def sthv1_image_name_formatter(x):
    return '{:05d}.jpg'.format(x)

def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'somethingv2',
        'somethingv1'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']
    if 'somethingv1' in dataset_name:
        formatter = sthv1_image_name_formatter
    elif 'somethingv2' in dataset_name:
        formatter = sthv2_image_name_formatter
    else:
        formatter = image_name_formatter
    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / '{}.hdf5'.format(video_id))

    if dataset_name == 'activitynet':
        training_data = ActivityNet(video_path,
                                    annotation_path,
                                    'training',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    video_loader=loader,
                                    video_path_formatter=video_path_formatter)
    else:
        training_data = VideoDataset(video_path,
                                     annotation_path,
                                     'training',
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)

    return training_data


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        input_type,
                        file_type,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'somethingv2',
        'somethingv1'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']

    if 'somethingv1' in dataset_name:
        formatter = sthv1_image_name_formatter
    elif 'somethingv2' in dataset_name:
        formatter = sthv2_image_name_formatter
    else:
        formatter = image_name_formatter

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / '{}.hdf5'.format(video_id))

    if dataset_name == 'activitynet':
        validation_data = ActivityNet(video_path,
                                      annotation_path,
                                      'validation',
                                      spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform,
                                      target_transform=target_transform,
                                      video_loader=loader,
                                      video_path_formatter=video_path_formatter)
    else:
        validation_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)

    return validation_data, collate_fn


def get_inference_data(video_path,
                       annotation_path,
                       dataset_name,
                       input_type,
                       file_type,
                       inference_subset,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'somethingv2'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']
    assert inference_subset in ['train', 'val', 'test']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / '{}.hdf5'.format(video_id))

    if inference_subset == 'train':
        subset = 'training'
    elif inference_subset == 'val':
        subset = 'validation'
    elif inference_subset == 'test':
        subset = 'testing'
    if dataset_name == 'activitynet':
        inference_data = ActivityNet(video_path,
                                     annotation_path,
                                     subset,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter,
                                     is_untrimmed_setting=True)
    else:
        inference_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter,
            target_type=['video_id', 'segment'])

    return inference_data, collate_fn
