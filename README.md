# Pytorch implementation of [A Style-Based Generator Architecture for Generative Adversarial Network](https://arxiv.org/abs/1812.04948)

## Requirements

- Python3
- Pytorch 1.0.0
- TensorBoardX

## Usage

train
```
python main.py 
    --config=path_to_config_file
```
Default configuration file is located in config directory.

## Currently completed task

* [x] Progressive method
* [x] Tuning
* [x] Add mapping and styles 
* [x] Remove traditional input 
* [x] Add noise inputs 
* [ ] Mixing regularization
