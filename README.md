# nisawa_work
This repository provides data and codes
# Contents
1. Instlation
2. Structure of folders and files
3. Citation
# Requirement
+ Python 3.7.10
+ tensprflow 2.2.0
# Installation
To get started with the project, follow these steps:
1. Clone the repository.
```
$ git clone https://github.com/nisawadaiki/nisawa_work.git
```

2.  Install the required packages from the requirements.txt file using conda:
```
$ pip install -r requirements.txt
```
3. Download each dataset and pretrained model, GTSRB(https://drive.google.com/drive/folders/1bDROZjdKLxUslbnUY7q0JbxQZhG-KSin?usp=drive_link) and GTSRB prtrained model() ImageNet() to the corresponding data folder under each dataset folder as the following structure.

4. Execute run.py of each CAM method as follows:
```
:$ python 
(default):$ python run.py [-device_num] [-data] [-mode] [--make_mask] [-mask_num] [-eval_sal] [--run_ins_del] [--run_adcc] [--hsv]
```
Option:
- device_num : int,default=0
  - use GPU device number : deep1 0 or 1 (default:0)
- data : string, default='GTSRB'
    - dataset
      - GTSRB
      - ImageNet
- 　mode: string, default='RaCF'
  - method: RaCF, RaCF_GradCAN, MC-RISE
  - evaluate:eval
- --make_mask:　指定しなければ作ったマスクをロード(指定でマスクを作成、保存)
- -mask_num: int ,default=5000
    - マスク数を決める　どのマスク数のフォルダを指定するかも含まれるので5000枚以外は必ず指定
-  -eval_sal : string ,default='RaCF'
    - -mode='eval'の時、どの手法を評価するかを決定  
- --run_ins_del: -mode='eval'の時、指定することで評価指標insertion,deletionを実行
- --run_adcc: -mode='eval'の時、指定することで評価指標adccを実行
- --hsv:指定することでGTSRBのデータをHSVに変換

# Structure of folders and files
```
.
├── run.py
├── util.py
├── make_saliency.py
├── make_maks.py
├── requirements.txt
├── method
│   ├── RaCF.py
│   ├── RaCFplusGardCAM.py
│   ├── MC_RISE.py
├── GTSRB
│   ├── code
│       ├── evaluate.py
│       ├── make_data.py
│       ├── vgg16.py
├── ImageNet
│   ├── code
│       ├── evaluate.py
│       ├── make_data.py

```
# RaCF
We used seismic event data provided by the Northern California Earthquake Data Center (NCEDC (2014)) and extracted 18,470 seismic events with a magnitude of larger than 3.0 recorded between 1970 and January 2022. The elapsed time τ is recorded in hours.
|| # of train | # of validation | # of test |
|---| ---- | ---- | ---- |
|# of events | 14,776   | 1,847         |  1,847  |
|# of windows | 14,746   | 1,817         |  1,817  |

To train or valid seq_rep_vec_Transformer_PP model, execute Main.py as follows :
```python
(train): $ python Main.py -gene=jisin --train
(valid): $ python Main.py -gene=jisin
```

# Police call event data
We used police call event data provided by the City and County of
San Francisco (San-Francisco (2021)). We extracted 55,262 events occurring at the first, second, and third most frequently recorded addresses, that is, 800 Block of BRYANT ST, 800 Block of MARKET ST, and 1000 Block of POTRERO AV, during 2003–2018 and created sliding windows for each address. For stratification, we selected windows whose last event τn+1 is recorded between 3:00 PM and 1:00 AM. The elapsed time is recorded in hours.
|| # of train | # of validation | # of test |
|---| ---- | ---- | ---- |
|# of events | 44,228 | 5,510 | 5,524  |
|# of windows |  24,314 | 3,006 | 2,965 |

To train or valid seq_rep_vec_Transformer_PP model, execute Main.py as follows :
```
first data
(train):$ python Main.py -gene=911_1_Address --train
(valid):$ python Main.py -gene=911_1_Address

second data
(train):$ python Main.py -gene=911_2_Address --train
(valid):$ python Main.py -gene=911_2_Address

third data
(train):$ python Main.py -gene=911_3_Address --train
(valid):$ python Main.py -gene=911_3_Address
```



# Citation
