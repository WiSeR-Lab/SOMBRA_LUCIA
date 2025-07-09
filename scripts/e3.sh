#! /usr/bin/bash

cd ..

model_name=(
    "AttentiveFusion"
    "CoAlign"
    "Where2comm"
    "V2VAM"
)

model_dir=(
    "./assets/opv2v_weights/attfusion"
    "./assets/opv2v_weights/coalign"
    "./assets/opv2v_weights/where2comm"
    "./assets/opv2v_weights/v2vam"
)

data_dir="./assets/traffic_jam_data/"

if [ ! -d "$data_dir" ]; then
    echo "Error: $data_dir does not exist, please download the data and extract it to $data_dir"
    echo "The data can be accesssed at https://zenodo.org/records/15523769"
    exit 1
fi

model_length=${#model_name[@]}

for ((i=0; i<model_length; i++)); do
    echo "Running attack on ${model_name[$i]} with SOMBRA..." 
    pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
        --data_dir ${data_dir} \
        --attack_mode mor --loss sombra \
        --save_perturb
done

for ((i=0; i<model_length; i++)); do
    echo "Running attack on ${model_name[$i]} with prior art..." 
    pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
        --data_dir ${data_dir} \
        --attack_mode mor --loss pa
done


echo "Moving perturbation to adv_feature for validating the attention scores"
mv ./assets/opv2v_weights/attfusion/perturb_iter10_lr0.1_mor_sombra ./assets/opv2v_weights/attfusion/adv_feature

echo "Validating the attention scores for different number of CAVs, this could take a while."
echo "Please check diff_cav.csv for results"
pixi run python case_study.py --model_dir ./assets/opv2v_weights/attfusion \
    --data_dir ${data_dir}