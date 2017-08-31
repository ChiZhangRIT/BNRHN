Batch Normalized Recurrent Neural Networks
===================

This experiment is implemented based on the image-to-text model described in the paper:

"Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge."

Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.

*IEEE transactions on pattern analysis and machine intelligence (2016).*

Full text available at: http://arxiv.org/abs/1609.06647

Show-and-tell model source code: https://github.com/tensorflow/models/tree/master/im2txt

## Usage

Prepare the training data and download the inception v3 model, see *im2txt/README*.

### Run the training script.
for regular LSTM
```
python train.py --input_file_pattern=data/mscoco/train-?????-of-00256 --inception_checkpoint_file=data/inception_v3/inception_v3.ckpt --train_dir=model/train --train_inception=false  --number_of_steps=1000000
```
for RHN
```
python train_rhn.py --input_file_pattern=data/mscoco/train-?????-of-00256 --inception_checkpoint_file=data/inception_v3/inception_v3.ckpt --train_dir=model/train_rhn --train_inception=false  --number_of_steps=1000000
```
for BNRHN
```
python train_bnrhn.py --input_file_pattern=data/mscoco/train-?????-of-00256 --inception_checkpoint_file=data/inception_v3/inception_v3.ckpt --train_dir=model/train_bnrhn --train_inception=false  --number_of_steps=400000
```

### Evaluation

Evaluation should be run concurrently with training so that summaries show up in TensorBoard.
```
export CUDA_VISIBLE_DEVICES=""
```
for regular LSTM
```
python evaluate.py --input_file_pattern=data/mscoco/val-?????-of-00004 --checkpoint_dir=model/train --eval_dir=model/eval
```
for RHN:
```
python evaluate_rhn.py --input_file_pattern=data/mscoco/val-?????-of-00004 --checkpoint_dir=model/train_rhn --eval_dir=model/eval_rhn
```
for BNRHN
```
python evaluate_bnrhn.py --input_file_pattern=data/mscoco/val-?????-of-00004 --checkpoint_dir=model/train_bnrhn --eval_dir=model/eval_bnrhn
```

Run a TensorBoard server in a separate process for real-time monitoring of training progress and evaluation metrics.
```
tensorboard --logdir=model
```
default: http://129.21.57.35:6006

### Fine tune the inception v3 model.
```
python train_bnrhn.py --input_file_pattern=data/mscoco/train-?????-of-00256 --train_dir=model/train_bnrhn --train_inception=true  --number_of_steps=3000000
```

### Generating captions
```
export CUDA_VISIBLE_DEVICES=""
python run_inference.py --checkpoint_path=model/train_bnrhn --vocab_file=data/mscoco/word_counts.txt --input_files=/cis/phd/cxz2081/data/mscoco/captioning/val2014/COCO_val2014_000000224477.jpg
```
Note: Use run_inference_KA.py to generate captions for image names in input_files. For example,
```
python run_inference_KA.py --checkpoint_path=model/train_bnrhn --vocab_file=data/mscoco/word_counts.txt --input_files=/cis/phd/cxz2081/data/KodakAlaris_ConsumerActivity/image_list.txt --output_file=/cis/phd/cxz2081/data/KodakAlaris_ConsumerActivity/image_captions.txt
```
Note: run_inference_KA.py is to generate captions for Kodak Alaris internal image dataset.

### Calculate scores (BLEU, METEOR, etc.)
run *cocoEvalCap.ipynb*

[Note]: This jupyter notebook actually consists of two parts:

1) *generate_captions.py*

2) *calc_metrics.py*

Instead of using jupyter notebook, we can run these two parts in python:

1) evaluate mscoco test dataset (generate captions for mscoco test dataset)

first change model in *inference_wrappr.py*, then modify inputs in *generate_captions.py*, and run:
```
python generate_captions.py
python generate_captions_for_selftest_images.py  # for "test" with known GT
```

2) extract image ids from TFRecords, generate captions, and calculate metrics
```
python calc_metrics.py  # image ids will be generated from TFRecords or loaded from file
```
for RHN (batch): modify line 88-93, line 179 in *evaluate_rhn.py*
