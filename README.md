
# Sensitivity-aware Distance Measurement for Boosting Metric Learning
Please follow the instruction to reproduce my experiments. Note that this project is based on [Powerful_benchmark](https://kevinmusgrave.github.io/powerful-benchmarker/).
## Installation
### Library
    torch >= 1.2
    pip install tensorboard    
    pip install scikit-learn    
    pip install matplotlib 
    pip install pandas  
    pip install ax-platform  
    pip install faiss-gpu
    pip install gdown
    pip install tqdm
    pip install munch
    pip install scipy
    pip install torchvision
### Prepare Dataset 

 - You can download CUB2011 from  [CUB2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
   Please extract the dataset to `data/cub2011/`
   
 - You can download Cars196 from [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).
   Please extract the dataset to `data/cars196/`

## Run
### Cars196
####  Example: Multi-similarity Loss
    # Train the model
    python run.py \
    --reproduce_results exp_configs/car_ms \
    --experiment_name car_ms \
    --merge_argparse_when_resuming \
    --split_manager~SWAP~1 {MLRCSplitManager: {}} \
    --config_dataset [default, with_cars196] \
	--config_general [default, with_cars196] 
    
    # Evaluation for 512-dim embedding
    python run.py \
    --reproduce_results exp_configs/car_ms \
    --experiment_name car_ms \
    --merge_argparse_when_resuming \
    --split_manager~SWAP~1 {MLRCSplitManager: {}} \
    --config_dataset [default, with_cars196] \
	--config_general [default, with_cars196] \
    --splits_to_eval [test] \
    --evaluate_ensemble
    
    # Evaluation for 128-dim embedding
    python run.py \
    --reproduce_results exp_configs/car_ms \
    --experiment_name car_ms \
    --merge_argparse_when_resuming \
    --split_manager~SWAP~1 {MLRCSplitManager: {}} \
    --config_dataset [default, with_cars196] \
	--config_general [default, with_cars196] \
    --splits_to_eval [test] \
    --evaluate
For other loss functions,  you can replace the config file path in `--reproduce_results`. Alternatives: `exp_configs/car_triplet`, `exp_configs/car_contrast`, `exp_configs/car_margin`.

### CUB2011
####  Example: Multi-similarity Loss

    # Train the model
    python run.py \
    --reproduce_results exp_configs/cub_ms \
    --experiment_name cub_ms \
    --merge_argparse_when_resuming \
    --split_manager~SWAP~1 {MLRCSplitManager: {}}
    
    # Evaluation for 512-dim embedding
    python run.py \
    --reproduce_results exp_configs/cub_ms \
    --experiment_name cub_ms \
    --merge_argparse_when_resuming \
    --split_manager~SWAP~1 {MLRCSplitManager: {}} \
    --splits_to_eval [test] \
    --evaluate_ensemble
    
    # Evaluation for 128-dim embedding
    python run.py \
    --reproduce_results exp_configs/cub_ms \
    --experiment_name cub_ms \
    --merge_argparse_when_resuming \
    --split_manager~SWAP~1 {MLRCSplitManager: {}} \
    --splits_to_eval [test] \
    --evaluate

For other loss functions, you can replace the config file path in `--reproduce_results`. Alternatives: `exp_configs/cub_triplet`, `exp_configs/cub_contrast`, `exp_configs/cub_margin`.

## Trained Models
We release our training logs and trained models in this table, please download from [Trained Models](https://drive.google.com/drive/folders/12KOmtKT47ZD0oJb92kW2qlOD2FSu6YFR?usp=sharing). 






|                      | Cars196                        |                         |                         |                            |                        |                        |
| :------------------: | :---------------------------: | :---------------------: | :---------------------: | :------------------------: | :--------------------: | :--------------------: |
|                      | Concatenated <br>  (512-dim)  |                         |                         | Separated <br>  (128-dim)  |                        |                        |
|                      | P@1                           | RP                      | MAP@R                   | P@1                        | RP                     | MAP@R                  |
| Pretrained           | 46\.89                        | 13\.77                  | 5\.91                   | 43\.27                     | 13\.37                 | 5\.64                  |
| NT-Xent             | 80\.99                        | 34\.96                  | 24\.40                  | 68\.16                     | 27\.66                 | 16\.78                 |
| ProxyNCA            | 83\.56                        | 35\.62                  | 25\.38                  | 73\.46                     | 28\.90                 | 18\.29                 |
| Margin/class        | 80\.04                        | 33\.78                  | 23\.11                  | 67\.54                     | 26\.68                 | 15\.88                 |
| N. Softmax          | 83\.16                        | 36\.20                  | 26\.00                  | 72\.55                     | 29\.35                 | 18\.73                 |
| CosFace             | 85\.52                        | 37\.32                  | 27\.57                  | 74\.67                     | 29\.01                 | 18\.80                 |
| ArcFace             | 85\.44                        | 37\.02                  | 27\.22                  | 72\.10                     | 27\.29                 | 17\.11                 |
| FastAP              | 78\.45                        | 33\.61                  | 23\.14                  | 65\.08                     | 26\.59                 | 15\.94                 |
| SNR~                 | 82\.02                        | 35\.22                  | 25\.03                  | 69\.69                     | 27\.55                 | 17\.13                 |
| MS+Miner            | 83\.67                        | 37\.08                  | 27\.01                  | 71\.80                     | 29\.44                 | 18\.86                 |
| SoftTriplet         | 84\.49                        | 37\.03                  | 27\.08                  | 73\.69                     | 29\.29                 | 18\.89                 |
| Contrast            | 81\.87                        | 35\.11                  | 24\.89                  | 69\.80                     | 27\.78                 | 17\.24                 |
| [**Contrast + Ours**](https://drive.google.com/drive/folders/19JgDWTNmIdRrdKYGLmo9A0tV_kAB1ynp?usp=sharing)           | **83\.56**                    | **36\.50**              | **26\.45**              | **73\.59**                 | **29\.77**             | **19\.31**             |
| Triplet             | 79\.13                        | 33\.71                  | 23\.02                  | 65\.68                     | 26\.67                 | 15\.82                 |
| [**Triplet + Ours**](https://drive.google.com/drive/folders/1u1eOqbcbFiY_SVaLXwCXRi64wHDxnCwE?usp=sharing)           | **81\.00**                    | **34\.76**              | **24\.24**              | **69\.60**                 | **28\.25**             | **17\.30**             |
| Margin              | 81\.16                        | 34\.82                  | 24\.21                  | 68\.24                     | 27\.25                 | 16\.40                 |
| [**Margin + Ours**](https://drive.google.com/drive/folders/1WKLKpw6aQVtMeq5vxfFbWCZ4B9IC7hW_?usp=sharing)           | **82\.25**                    | **35\.49**              | **25\.11**              | **71\.99**                 | **28\.79**             | **18\.07**             |
| MS                  | 85\.14                        | 38\.09                  | 28\.07                  | 73\.77                     | 29\.92                 | 19\.32                 |
| [**MS + Ours**](https://drive.google.com/drive/folders/1ZX-ISrWjMZp-xw6qMPKbT-Lw4ojymqao?usp=sharing)           | **86\.66**                    | **39\.39**              | **29\.67**              | **77\.19**                 | **31\.67**             | **21\.28**             |                


---


|                      | CUB2011                       |                      |                                |                             |                         |                        |
| :------------------: | :---------------------------: | :------------------: | :----------------------------: | :-------------------------: | :---------------------: | :--------------------: |
|                      | Concatenated <br>  (512 dim)  |                      |                                | Separated <br>   (128-dim)  |                         |                        |
|                      | P@1                           | RP                   | MAP@R                          | P@1                         | RP                      | MAP@R                  |
| Pretrained           | 51\.05                        | 24\.85               | 14\.21                         | 50\.54                      | 25\.12                  | 14\.53                 |
| NT-Xent             | 66\.61                        | 35\.96               | 25\.09                         | 58\.12                      | 30\.81                  | 19\.87                 |
| ProxyNCA            | 65\.69                        | 35\.14               | 24\.21                         | 57\.88                      | 30\.16                  | 19\.32                 |
| Margin/class        | 64\.37                        | 34\.59               | 23\.71                         | 55\.56                      | 29\.32                  | 18\.51                 |
| N. Softmax          | 65\.65                        | 35\.99               | 25\.25                         | 58\.75                      | 31\.75                  | 20\.96                 |
| CosFace             | 67\.32                        | 37\.49               | 26\.70                         | 59\.63                      | 31\.99                  | 21\.21                 |
| ArcFace             | 67\.50                        | 37\.31               | 26\.45                         | 60\.17                      | 32\.37                  | 21\.49                 |
| FastAP              | 63\.17                        | 34\.20               | 23\.53                         | 55\.58                      | 29\.72                  | 19\.09                 |
| SNR                 | 66\.44                        | 36\.56               | 25\.75                         | 58\.06                      | 31\.21                  | 20\.43                 |
| MS+Miner            | 67\.73                        | 37\.37               | 26\.52                         | 59\.41                      | 31\.93                  | 21\.01                 |
| SoftTriplet         | 67\.27                        | 37\.34               | 26\.51                         | 59\.94                      | 32\.12                  | 21\.31                 |
| Contrast            | 68\.13                        | 37\.24               | 26\.53                         | 59\.73                      | 31\.98                  | 21\.18                 |
| [**Contrast + Ours**](https://drive.google.com/drive/folders/1il_6vNk8NXGRRsZJ6Ht8bA_F3XGtpMip?usp=sharing)           | **68\.96**                    | **37\.60**           | **26\.88**                     | **60\.24**                  | **32\.54**              | **21\.67**             |
| Triplet             | 64\.24                        | 34\.55               | 23\.69                         | 55\.76                      | 29\.55                  | 18\.75                 |
| [**Triplet + Ours**](https://drive.google.com/drive/folders/1VG_vsS1wKQOHStfdfPoDgqpZTGrhf3x0?usp=sharing)           | **65\.46**                    | **35\.00**           | **24\.20**                     | **57\.87**                  | **30\.68**              | **19\.85**             |
| Margin              | 63\.60                        | 33\.94               | 23\.09                         | 54\.78                      | 28\.86                  | 18\.11                 |
| [**Margin + Ours**](https://drive.google.com/drive/folders/13-MEOH-FCJgRF8Bagpd3ZaIpfwIoRwdS?usp=sharing)           | **64\.24**                    | **34\.33**           | **23\.47**                     | **56\.50**                  | **29\.61**              | **18\.80**             |
| MS                  | 65\.04                        | 35\.40               | 24\.70                         | 57\.60                      | 30\.84                  | 20\.15                 |
| [**MS + Ours**](https://drive.google.com/drive/folders/16ZMaOaDxidJ95pp18NDPhdHgb7VmK3kQ?usp=sharing)           | **67\.08**                    | **36\.23**           | **25\.61**                     | **59\.22**                  | **31\.82**              | **21\.24**             |
