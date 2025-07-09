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


model_length=${#model_name[@]}

for ((i=0; i<model_length; i++)); do
        echo "Running TOR attack on ${model_name[$i]} with SOMBRA... BUT with LUCIA enabled!" 
        pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
            --data_dir ${data_dir} \
            --attack_mode tor --target_id random --loss sombra \
            --defense
done

for ((i=0; i<model_length; i++)); do
        echo "Running TOR attack on ${model_name[$i]} with prior art... BUT with LUCIA enabled!" 
        pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
            --data_dir ${data_dir} \
            --attack_mode tor --target_id random --loss pa \
            --defense
done

for ((i=0; i<model_length; i++)); do
    echo "Running MOR attack on ${model_name[$i]} with SOMBRA... BUT with LUCIA enabled!" 
    pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
        --data_dir ${data_dir} \
        --attack_mode mor --loss sombra \
        --defense
done

for ((i=0; i<model_length; i++)); do
    echo "Running MOR attack on ${model_name[$i]} with prior art... BUT with LUCIA enabled!" 
    pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
        --data_dir ${data_dir} \
        --attack_mode mor --loss pa \
        --defense
done

for ((i=0; i<model_length; i++)); do
        echo "Running TOR attack on ${model_name[$i]} with SOMBRA... BUT with ROBOSAC enabled!" 
        pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
            --data_dir ${data_dir} \
            --attack_mode tor --target_id random --loss sombra \
            --robosac
done

for ((i=0; i<model_length; i++)); do
        echo "Running TOR attack on ${model_name[$i]} with prior art... BUT with ROBOSAC enabled!" 
        pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
            --data_dir ${data_dir} \
            --attack_mode tor --target_id random --loss pa \
            --robosac
done

for ((i=0; i<model_length; i++)); do
    echo "Running MOR attack on ${model_name[$i]} with SOMBRA... BUT with ROBOSAC enabled!" 
    pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
        --data_dir ${data_dir} \
        --attack_mode mor --loss sombra \
        --robosac
done

for ((i=0; i<model_length; i++)); do
    echo "Running MOR attack on ${model_name[$i]} with prior art... BUT with ROBOSAC enabled!" 
    pixi run python cp_attack.py --model ${model_name[$i]} --model_dir ${model_dir[$i]} \
        --data_dir ${data_dir} \
        --attack_mode mor --loss pa \
        --robosac
done