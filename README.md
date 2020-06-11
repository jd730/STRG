# Videos as Space-Time Region Graph

## Summary

This is the unofficial PyTorch code for the following paper:

[
Wang, Xiaolong, and Abhinav Gupta. "Videos as space-time region graphs." Proceedings of the European conference on computer vision (ECCV). 2018.
](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaolong_Wang_Videos_as_Space-Time_ECCV_2018_paper.pdf)


## Requirements

* [PyTorch](http://pytorch.org/) (ver. 1.2+ required)
* [Torchvision](http://pytorch.org/) (ver. 0.4+ required)

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c soumith
```

```bash
pip install -r requirements.txt
```


* FFmpeg, FFprobe

* Python 3

## Preparation

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs mp4_video_dir_path jpg_video_dir_path kinetics
```

* Generate annotation file in json format similar to ActivityNet using ```util_scripts/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python -m util_scripts.kinetics_json csv_dir_path 700 jpg_video_dir_path jpg dst_json_path
```


### Something-Something v1/v2

* Download videos from the official [website](https://20bn.com/datasets/something-something/v2#download).
* For Something-Something v2, please run `util_scripts/vid2img_sthv1.[py`

```bash
python util_scripts/sthv1_json.py 'data/something/v1' 'data/something/v1/img' 'data/sthv1.json'
```

```bash
python util_scripts/sthv2_json.py 'data/something/v2' 'data/something/v2/img' 'data/sthv2.json'
```



## Running the code

Assume the structure of data directories is the following:

```misc
~/
  data/
    something/
      v1/
        img/
          .../ (directories of video names)
            ... (jpg files)
      v2/
        img/
          .../ (directories of video names)
            ... (jpg files)
    kinetics_videos/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      save_100.pth
    kinetics.json
```

Confirm all options.

```bash
python main.py -h
```

Train ResNets-50 on the Kinetics-700 dataset (700 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5
```


Calculate top-5 class probabilities of each video using a trained model (~/data/results/save_200.pth.)  
Note that ```inference_batch_size``` should be small because actual batch size is calculated by ```inference_batch_size * (n_video_frames / inference_stride)```.

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

Evaluate top-1 video accuracy of a recognition result (data/results/val.json).

```bash
python -m util_scripts.eval_accuracy data/sthv2.json data/results/val.json --subset val -k 1 --ignore
```

Fine-tune fc layers of a pretrained model (~/data/models/resnet-50-kinetics.pth) on UCF-101.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 101 --n_pretrain_classes 700 \
--pretrain_path models/resnet-50-kinetics.pth --ft_begin_module fc \
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```
