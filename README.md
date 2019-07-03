# Pytorch implementation of [A Style-Based Generator Architecture for Generative Adversarial Network](https://arxiv.org/abs/1812.04948)

# IMPORTANT NOTES (7/2)

Previously, style_mapper did not work properly. It just feed z to generator. 

Now, generator was fixed. I train model again to get a good results. T_T

## Requirements

- Python3
- Pytorch > 1.0.0
- TensorBoardX
- fire
- apex [optional] 

I recommend install apex. apex.amp improves memory efficiency and learning speed using [mixed precision](https://arxiv.org/abs/1710.03740).

But you do not need to install it if you do not want it.

## Usage

train
```
python main.py 
    --config_file=path_to_config_file
    --checkpoint=path_to_config_file[default='']
```

inference
```
python main.py 
    --config_file=path_to_config_file
    --run_type=inference
```

Default configuration file is located in config directory.

## Currently completed task

* [x] Progressive method
* [x] Tuning
* [x] Add mapping and styles 
* [x] Remove traditional input 
* [x] Add noise inputs 
* [x] Mixing regularization

## Fake image and real image score graph

![graph](images/graph.png)

## Inference Images
#### NOTE: These images does not use style mapping network. (just using z, the normal distribution) I still training for upload style mapped images. It will be updated 256x256 images with pretrained checkpoints. (I do not have an environment to train high resolution images ... please contribute!)

### 8x8 images
![8x8](images/8x8.png)
### 16x16 images
![16x16](images/16x16.png)
### 32x32 images
![32x32](images/32x32.png)
### 64x64 images
![64x64](images/64x64.png)
### 128x128 images
![128x128](images/128x128.png)
