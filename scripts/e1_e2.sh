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

data_dir="./assets/test/"

target_id=(
    "random"
    "in"
    "out"
)

model_length=${#model_name[@]}
target_id_length=${#target_id[@]}

for ((i=0; i<model_length; i++)); do
    for ((j=0; j<target_id_length; j++)); do
        echo "Running TOR attack on ${model_name[$i]} with SOMBRA..." 
        echo "Target removing mode: ${target_id[$j]}"
        pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
            --data_dir ${data_dir} \
            --attack_mode tor --target_id ${target_id[$j]} --loss sombra
    done
done

for ((i=0; i<model_length; i++)); do
    for ((j=0; j<target_id_length; j++)); do
        echo "Running TOR attack on ${model_name[$i]} with prior art..." 
        echo "Target removing mode: ${target_id[$j]}"
        pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
            --data_dir ${data_dir} \
            --attack_mode tor --target_id ${target_id[$j]} --loss pa
    done
done

for ((i=0; i<model_length; i++)); do
    echo "Running MOR attack on ${model_name[$i]} with SOMBRA..." 
    pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
        --data_dir ${data_dir} \
        --attack_mode mor --loss sombra
done

for ((i=0; i<model_length; i++)); do
    echo "Running MOR attack on ${model_name[$i]} with prior art..." 
    pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
        --data_dir ${data_dir} \
        --attack_mode mor --loss pa
done