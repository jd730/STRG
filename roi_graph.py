import os
import pickle

import numpy as np

import torch
import torch.nn as n
import torch.nn.functional as F
import pdb


def extract_rois(rois_features, rois, option='first'):
    return_features = np.zeros((0, rois_features[0].shape[1]))
    return_rois = np.zeros((0, 5))
    connection_info = []
    pdb.set_trace()
    for i, (f, r) in enumerate(zip(rois_features, rois)):
        is_np = type(r) is np.ndarray
        if option == 'first':
            target = 0
        else :
            target = np.random.randint(2)

        idx = (r[:,0] == target)
        extracted_feature = f[idx]
        extracted_rois = r[idx]

        if len(extracted_rois) == 0:
            prev_rois = np.zeros((0,5))
            continue

        # change index
        extracted_rois[:,0] = i
        connection_info = calculate_spatial_iou(extracted_rois, connection_info, is_np)
        if i > 0 :
            connection_info = calculate_temporal_iou(extracted_rois, prev_rois, \
                    connection_info, is_np)

        prev_rois = extracted_rois

        if not is_np:
            extracted_feature = extracted_feature.cpu().numpy()
            extracted_rois = extracted_rois.cpu().numpy()

        return_features = np.vstack((return_features, extracted_feature))
        return_rois = np.vstack((return_rois, extracted_rois))

    connection_info = [sorted(e) for e in connection_info]
    return return_features, return_rois, connection_info


def calculate_spatial_iou(rois, overlapped_idx=[], is_np=True):
    area = (rois[:,4] - rois[:,2] + 1) * (rois[:,3] - rois[:,1] + 1)
    start_idx = len(overlapped_idx)
    for i in range(len(rois)-1):
        iou = get_iou(rois[i], rois[i+1:], area[i], area[i+1:], is_np)
        iou_idx = iou2idx(iou, start_idx + i+1, is_np)
        overlapped_idx.append(iou_idx)

    overlapped_idx.append([])
    for i in reversed(range(start_idx, len(overlapped_idx))):
        iou_idx = overlapped_idx[i]
        for prev_idx in iou_idx : # reverse process
            overlapped_idx[prev_idx].append(i)

    return overlapped_idx


def calculate_temporal_iou(rois, prev_rois, overlapped_idx=[], is_np=True):
    if len(prev_rois) == 0:
        return overlapped_idx

    area = (rois[:,4] - rois[:,2] + 1) * (rois[:,3] - rois[:,1] + 1)
    prev_area = (prev_rois[:,4] - prev_rois[:,2] + 1) * (prev_rois[:,3] - prev_rois[:,1] + 1)
    current_start_idx = len(overlapped_idx) - len(rois)
    prev_start_idx =  current_start_idx - len(prev_rois)

    for i, (a, r) in enumerate(zip(area, rois)):
        iou = get_iou(r, prev_rois, a, prev_area, is_np)
        iou_idx = iou2idx(iou, prev_start_idx, is_np)


        overlapped_idx[i+current_start_idx] += iou_idx
        for prev_idx in iou_idx:
            overlapped_idx[prev_idx].append(i + current_start_idx)

    return overlapped_idx


def iou2idx(iou, start_idx=0, is_np=True, threshold=0):
    iou_idx = iou > 0
    if is_np:
        iou_idx = np.nonzero(iou_idx)[0] + start_idx
    else :
        iou_idx = torch.nonzero(iou_idx) + start_idx
        iou_idx = iou_idx.cpu()
    iou_idx = iou_idx.reshape(-1)
    iou_idx = iou_idx.tolist()

    return iou_idx


def get_iou(roi, rois, area, areas, is_np=True) :
    if is_np:
        y_min = np.maximum(roi[1], rois[:,1])
        x_min = np.maximum(roi[2], rois[:,2])
        y_max = np.minimum(roi[3], rois[:,3])
        x_max = np.minimum(roi[4], rois[:,4])
        intersection = np.maximum(0, x_max - x_min + 1) * np.maximum(0, y_max - y_min + 1)
    else:
        y_min = torch.max(roi[1], rois[:,1])
        x_min = torch.max(roi[2], rois[:,2])
        y_max = torch.min(roi[3], rois[:,3])
        x_max = torch.min(roi[4], rois[:,4])
        axis0 = x_max - x_min + 1
        axis1 = y_max - y_min + 1
        axis0[axis0 < 0] = 0
        axis1[axis1 < 0] = 0
        intersection = axis0 * axis1
    iou = intersection / (areas + area - intersection)
    return iou


def convert2batch(rois_features, rois, rois_graph_all):
    N = len(rois_features)
    N_rois = [len(r) for r in rois]
    max_N_rois = np.max(N_rois)

    ret_features = torch.zeros((N, max_N_rois, rois_features[0].shape[-1]))
    ret_rois = -torch.ones((N, max_N_rois, 5))

    for i, j in zip(range(N), N_rois):
        if j > 0:
            ret_features[i][:j] = rois_features[i]
            ret_rois[i][:j] = rois[i]

    return ret_features, ret_rois, rois_graph_all, N_rois #, N_connection


def get_st_graph(rois, connection):
    is_np = type(rois) is np.ndarray

    N = len(rois)
    if is_np:
        st_graph = np.zeros((N,N))
    else:
        st_graph = torch.zeros((N,N))
#        if 'cuda' in rois.device():
#            st_graph = st_graph.cuda()

    if N ==0 :
        return st_graph

    area = (rois[:,4] - rois[:,2] + 1) * (rois[:,3] - rois[:,1] + 1)
    for i in range(N):
        c = connection[i]
        if len(c) == 0:
            continue
        if type(c) is list:
            if is_np:
                idx = np.asarray(c)
            else:
                idx = torch.LongTensor(c)
        elif is_np:
            c_idx = np.nonzero(c > 0)[0]
            if len(c_idx) == 0:
                continue
            idx = np.asarray(c[c_idx])
        else:
            c_idx = torch.nonzero(c > 0).view(-1)
            if len(c_idx) == 0:
                continue
            idx = c[c_idx].long()

        ious = get_iou(rois[i], rois[idx], area[i], area[idx], is_np=is_np)
        st_graph[i, idx] = ious / ious.sum()
        if (is_np and np.isnan(st_graph[i,idx]).sum() > 0) or (not is_np and torch.isnan(st_graph[i,idx]).sum() > 0 ):
            pdb.set_trace()

    return st_graph



def front_filter(rois, connection_info):
    current_idx = []
    current = rois[0,0]
    # getting frame change points
    for i,r in enumerate(rois):
        if r[0] != current:
            current_idx.append(i)
            current = r[0]
    if rois[-1,0] != current:
        current_idx.append(len(rois)-1)

    current_idx.append(len(rois)) # dummy
    pos = 0
    for i in range(len(rois)):
        connection_info[i] = list(filter(lambda x: x >= current_idx[pos], connection_info[i]))
        if i  == current_idx[pos]:
            pos += 1

    return connection_info



def get_roi_graph(rois_features, rois, connection_info, st=0, ed=1, st_graph=None, is_np=False, sample_rois=-1):
    candidate = np.logical_and(rois[:,0] >= st, rois[:,0] < ed)
    candidate_idx = np.nonzero(candidate)[0]
    if len(candidate_idx) == 0 :
        if is_np:
            return np.zeros((0, rois_features.shape[1])), np.zeros((0,5)), [[]]
        else:
            return torch.zeros((0, rois_features.shape[1])), torch.zeros((0,5)), [[]]
    st_idx = np.min(candidate_idx)
    ed_idx = np.max(candidate_idx) + 1


    if sample_rois > 0 and sample_rois < ed_idx - st_idx :
        sample_idx = np.sort(np.random.randint(st_idx, ed_idx, sample_rois))
        converter = {}
        for i, c in enumerate(sample_idx):
            converter[c] = i
    else :
        sample_idx = None

    if st_graph is None:
        if sample_idx is not None :
            return_connection = np.asarray(connection_info)
            return_features = rois_features[sample_idx]
            return_rois = rois[sample_idx]
            return_connection = return_connection[sample_idx]
            for i in range(len(return_rois)):
                return_connection[i] = list(filter(lambda x : x in sample_idx, return_connection[i]))
                return_connection[i] = list(map(lambda x: converter[x], return_connection[i]))
        else:
            return_features = rois_features[st_idx:ed_idx]
            return_rois = rois[st_idx:ed_idx]
            return_connection = connection_info[st_idx:ed_idx]
            # filtering connected with outer proposals
            for i in range(len(return_rois)):
                if return_rois[i,0] > st :
                    break
                return_connection[i] = list(filter(lambda x: x >= st_idx, return_connection[i]))
            for i in reversed(range(len(return_rois))):
                if return_rois[i,0] < ed-1:
                    break
                return_connection[i] = list(filter(lambda x: x < ed_idx, return_connection[i]))
            # change index
            for i in range(len(return_rois)):
                return_connection[i] = list(map(lambda x: x-st_idx, return_connection[i]))

    if not is_np:
        return_features = torch.FloatTensor(return_features)
        return_rois = torch.FloatTensor(return_rois)

    if st_graph is None:
        return_st_graph = get_st_graph(return_rois, return_connection)
    else:
        return_st_graph = st_graph[st_idx:ed_idx][:,st_idx:ed_idx]

    del connection_info, return_connection
    return return_features, return_rois, return_st_graph



if __name__ == '__main__':
    out = pickle.load(open('/data3/dataset/UCF/I3D_feature_extraction/THUMOS/train_rgb_box/video_validation_0000904.pkl','rb'))
    rois_features, rois, connection_info = sample_rois(out[0], out[1])
    out = get_roi_graph(rois_features, rois, connection_info, st=100, ed=150)

    N = len(rois)

    pdb.set_trace()


