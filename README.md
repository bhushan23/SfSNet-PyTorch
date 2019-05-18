# SfSNet-PyTorch

## Source Description
```
├─── data_loading.py: data loading and pre-processing methods <br>
├─── generate_dataset_csv.py: generate csv given dataset directory <br>
├─── interpolate.py: Create interpolation result from provided image directory <br>
├─── main_gen_pseudo-data.py: Train Skip-Connection based network on Synthetic dataset and generate Pseudo-Supervision data for CelebA dataset<br>
├─── main_gen_synthetic_and_full.py: Train on synthetic data, generate pseudo-supervision data, train on mix data<br>
├─── main_mix_training.py: Train SfSNet on mix data. Need to provide both CelebA and Synthetic dataset directory<br>
├─── models.py: Definition of all the models used. Skip-Net and SfSNet <br>
├─── shading.py: Shading generation method from Normal and Spherical Harmonics <br>
├─── train.py: Train and test rountines <br>
├─── utils.py: Help rountines <br>
├─── data/: sample small scale dataset to be used. Use ON_SERVER=False in main_* scripts to use this dataset <br>
├─── pretrained/: pretrained model provided by SfSNet Author- 'load_model_from_pretrained' loads this model's weights to our model<br>
```

## Usage of main_* to train the model
```
usage: main_mix_training.py [-h] [--batch_size N] [--epochs N] [--lr LR]
                            [--wt_decay W] [--no_cuda] [--seed S]
                            [--read_first READ_FIRST] [--details DETAILS]
                            [--load_pretrained_model LOAD_PRETRAINED_MODEL]
                            [--syn_data SYN_DATA] [--celeba_data CELEBA_DATA]
                            [--log_dir LOG_DIR] [--load_model LOAD_MODEL]

SfSNet - Residual

optional arguments:
  -h, --help            show this help message and exit
  --batch_size N        input batch size for training (default: 8)
  --epochs N            number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.001)
  --wt_decay W          SGD momentum (default: 0.0005)
  --no_cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --read_first READ_FIRST
                        read first n rows (default: -1) from the dataset
                        This is helpful to load part of the data. Note that, internally
                        we change this to sample randomly with seed value 100
  --details DETAILS     Explaination of the run
                        String provided will be written into root log directory
                        We perform many experiments and then get lost on the results and what was this experiment for.
                        This txt file will help us understand what was the purpose of this experiment
  --load_pretrained_model LOAD_PRETRAINED_MODEL
                        Pretrained model path for SfSNet based model provided by author
  --syn_data SYN_DATA   Synthetic Dataset path directory high level containing csv files
  --celeba_data CELEBA_DATA
                        CelebA Dataset path high level containing train, test folder and csv files
  --log_dir LOG_DIR     Log Path
                        Where to log and store results, model
  --load_model LOAD_MODEL
                        load model from following directory
```

Example
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --epochs 100 --lr 0.0002 --batch_size 8 --read_first 10000
--log_dir ./results/skip_net/exp4/ --details 'Skipnet with normals'
```
