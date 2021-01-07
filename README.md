# Videos as Space-Time Region Graph

## Summary

* This repository is for testing the idea of the following paper:

[
Wang, Xiaolong, and Abhinav Gupta. "Videos as space-time region graphs." Proceedings of the European conference on computer vision (ECCV). 2018.
](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaolong_Wang_Videos_as_Space-Time_ECCV_2018_paper.pdf)

* It means that it may contain several mismatch with the original implementation introduced on the paper.

* Also the performance is much lower than the publication (24 vs 43) and I never test Kinetics pre-trained ResNet-50-I3D.

## Notes

* This repository is based on https://github.com/kenshohara/3D-ResNets-PyTorch.

* The architecture of ResNet-50-I3D in the paper is different from that in the above repository. I did not use Kinetics pre-trained model but use ImageNet pre-trained model.

* Currently, RPN is used on every iteration which requires approximately 3 times more training time.

* Kinetics pre-trained model can be found in [here](https://github.com/joaanna/something_else).


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

### Data Path

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

### Kinetics Pre-training

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

### Something-Something-v1

First of all, we need to train backbone network (ResNet-50-I3D) for 100 epochs with learning rate as 0.00125 (decayed at 90 epoch to 0.000125)
The original batchsize is 8 but in this implementation, we use 32 to reduce the training time.

```bash
python main.py --root_path data --video_path data/something/v1/img --annotation_path sthv1.json \
--result_path resnet_strg_imgnet_bs32 --dataset somethingv1 --n_classes 174 --n_pretrain_classes 700 \
--ft_begin_module fc --tensorboard --wandb --conv1_t_size 5 --learning_rate 0.00125 --sample_duration 32 \
--n_epochs 100 --multistep_milestones 90 --model resnet_strg --model_depth 50 --batch_size 32 \
--n_threads 8 --checkpoint 1
```

Then, we need to train with GCN module until 30 epochs with learning rate as 0.000125.

```bash
python main.py --root_path data --video_path data/something/v1/img --annotation_path sthv1.json \
--result_path resnet_strg_imgnet_32_gcn --dataset somethingv1 --n_classes 174 --n_pretrain_classes 174 \
--ft_begin_module fc --tensorboard --wandb --conv1_t_size 5  --learning_rate 0.000125 \
--sample_duration 32 --n_epochs 30 --model resnet_strg --model_depth 50 --batch_size 32 \
--nrois 10 --det_interval 2 --strg \
--n_threads 8 --checkpoint 1 --pretrain_path resnet_strg_imgnet_bs32/save_100.pth
```

## Results on Something-Something-v1

### The published results

| Model name         | ResNet-50-I3D | ResNet-50-I3D + STRG |
| ------------------ |---------------- | -------------- |
| Top-1 Accuracy   |     41.6%         |      43.3% |


### This repo results (without using Kinetic pretraining model)

| Model name         | ResNet-50-I3D | ResNet-50-I3D + STRG |
| ------------------ |---------------- | -------------- |
| Top-1 Accuracy   |     23.2%         |      24.5% |



