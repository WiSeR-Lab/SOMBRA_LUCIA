# From Threat to Trust: Exploiting Attention Mechanisms for Attacks and Defenses in Cooperative Perception

This repo contains the official implementation for the paper, including the attack SOMBRA and the defense LUCIA.

## Dataset and Model Download

Please visit the official website of [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) for latest dataset download instructions. Our evaluation is conducted on the test split of the data.

Pretrained model weights can be downloaded from [OpenCOOD repo](https://github.com/DerrickXuNu/OpenCOOD/tree/main). We included Attentive Fusion, CoAlign, Where2comm, and V2VAM in our evaluation.

_Note:_ Alternatively, you can download the OPV2V test split [here](https://arizona.box.com/s/kd9gisu2xaiw7s4fpxzqpnag4md18zhd) (19.6G) and the pre-trained model weights [here](https://arizona.box.com/s/kdcvfnqapcln931y7jjetw589sf81rtj) (114MB). You can also use `./scripts/data_model_download.sh` to download and extract the data and model weights.

The dataset after decompression has the following structure:

```
test
|-- 2021_08_18_19_48_05
|   |---- data_protocol.yaml
|   |---- 1045
|   |---- 1054
|...
|-- scenario_timestamp_id
|   |---- data_protocol.yaml
|   |---- agent_id_1
|   |---- ...
|   |---- agent_id_n
```

The model weights after decompression has the following structure:

```
opv2v_weights
|-- attfusion
|   |---- latest.pth
|   |---- config.yaml
|-- coalign
|   |---- net_epoch15.pth
|   |---- config.yaml
|-- v2vam
|   |---- latest.pth
|   |---- config.yaml
|-- where2comm
|   |---- net_epoch50.pth
|   |---- config.yaml
```

## Environment Setup

We use `pixi` for easier and faster env setup. More information can be found at [here](https://pixi.sh). 

We tested on CUDA 11.8 / 12.0. Please edit the `pixi.toml` file to change the `pytorch-cuda` version and `spconv-cu118` version accordingly (e.g., `spconv-cu120`).

Next, with `pixi` installed, simply run the following command to get the package installed (if not yet) in the virtual environment and activate it (to deactivate simply run `exit`).

```
pixi shell
```

Finally, setup and build the dependencies for OpenCOOD and NMS GPU version using the following commands:

```
pixi run opencood_setup
pixi run nms_gpu_build
```

Alternatively, you can run

```
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
```

## Reproducing results

We have put evaluation scripts in `./scripts/` where `e1_e2.sh` is used for evaluating the TOR and MOR attack using SOMBRA (ours) and prior art; `e3.sh` is for case study on the traffic jam scenario; and `e4.sh` is for evaluating the defense LUCIA (ours) and ROBOSAC.

## Evaluation

For evaluation, use `python cp_attack.py` with corresponding arguments. Use `--help` arguments to show all available arguments. Use `--loss sombra` for our attack SOMBRA, `--loss pa` for attack using the loss from prior art, `--loss bim` for untargeted attack. Specify `--defense` for LUCIA and `--robosac` for ROBOSAC defense.

For targeted object removal, you can specify the target object using `--target_id` followed by corresponding object ID in the dataset, or `in/out` for randomly sampled target within/beyond victim's Line-of-Sight, or `random` for just, a randomly sampled target.

Example:
```
python cp_attack.py --model_dir <path_to_model, e.g. attfusion> --model AttentiveFusion --data_dir <path_to_opv2v_test> --attack_mode mor --loss sombra (--defense)
```

The detailed attack results would be saved under the same folder as the model weight.

## Case Study

For the traffic jam case study, the latest link to the dataset can be accessed on [our GitHub repo page](https://github.com/orgs/WiSeR-Lab/SOMBRA_LUCIA).

The evaluation is done in two parts to save time on perturbation generation. 

First, run 

```
python cp_attack.py --model_dir <path_to_model, e.g. attfusion> --model <Model_Name> --data_dir <Path_to_Traffic_Jam> --attack_mode mor --loss sombra --save_perturb
```

to save the perturbation generated with knowledge of only the attacker and victim's features.

Next, rename the the folder that stores the pertubed attacker feature as `adv_feature`, and runs the following

```
python case_study.py --model_dir <path_to_model, e.g. attfusion> --data_dir <Path_to_Traffic_Jam> 
```



