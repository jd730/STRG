import argparse
import json
from pathlib import Path

import pandas as pd

from utils import get_n_frames
import pdb
import tqdm
import json


def convert_json_to_dict(csv_path, subset):
    lines = json.load(open(csv_path,'r'))
    database = {}

    for line in lines:
        video_id = line['id']
        database[video_id] = {}
        database[video_id]['subset'] = subset
        if subset != 'testing':
            label = line['template'].replace('[','').replace(']','')
            database[video_id]['annotations'] = {'label': label}
        else:
            database[video_id]['annotations'] = {}

    return database



def convert_csv_to_dict(csv_path, subset):
    lines = open(csv_path, 'r').readlines()
    keys = []
    key_labels = []
    database = {}

    for line in lines:
        video_id, nframe, label = line.strip('\n').split(' ')

        database[video_id] = {}
        database[video_id]['subset'] = subset
        if subset != 'testing':
            database[video_id]['annotations'] = {'label': label}
        else:
            database[video_id]['annotations'] = {}

    return database


def load_labels(train_csv_path):
    data = open(train_csv_path, 'r').readlines()
    data = [e.strip('\n') for e in data]
    return data
#    data = pd.read_csv(train_csv_path, header=None)
#    return data.iloc[:, 0].tolist()


def convert_sthv2_csv_to_json(class_file_path, train_csv_path, val_csv_path,
                            test_csv_path, video_dir_path, dst_json_path):
    labels = load_labels(class_file_path)
    train_database = convert_json_to_dict(train_csv_path, 'training')
    val_database = convert_json_to_dict(val_csv_path, 'validation')
    if test_csv_path.exists():
        test_database = convert_json_to_dict(test_csv_path, 'testing')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    if test_csv_path.exists():
        dst_data['database'].update(test_database)

    count = 0
    for k, v in tqdm.tqdm(dst_data['database'].items()):
        if 'label' in v['annotations']:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = video_dir_path / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)
        v['video_path'] = str(video_path)
#        count += 1
#        if count == 1000:
#            break

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path',
        default='data/something/v2',
        type=Path,
        help=('Directory path including moments_categories.txt, '
              'trainingSet.csv, validationSet.csv, '
              '(testingSet.csv (optional))'))
    parser.add_argument('video_path',
        default='data/something/v2/img',
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('dst_path',
                        default='./',
                        type=Path,
                        help='Path of dst json file.')

    args = parser.parse_args()

    class_file_path = args.dir_path / 'category.txt'
    train_csv_path = args.dir_path / 'something-something-v2-train.json'
    val_csv_path = args.dir_path / 'something-something-v2-validation.json'
    test_csv_path = args.dir_path / 'something-something-v2-test.json'
#    train_csv_path = args.dir_path / 'train_videofolder.txt'
#    val_csv_path = args.dir_path / 'val_videofolder.txt'
#    test_csv_path = args.dir_path / 'test_videofolder.txt'

    convert_sthv2_csv_to_json(class_file_path, train_csv_path, val_csv_path,
                            test_csv_path, args.video_path, args.dst_path)
