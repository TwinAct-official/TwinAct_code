#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PORT=24001


TRAIN_PATH="
    --pretrained_model_name_or_path="./ckpt/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path="./ckpt/sdxl-vae-fp16-fix" \
    --train_data_dir="./dataset" \
    --caption_column="text" \
    --output_dir="./exp/" \
    --report_to="wandb"
"
placeholder_tokens="
    <a1>,<a2>,<a3>
"

initializer_tokens="
    Action,action,action
"

VALID_PROMPT="
A polar bear <a1> <a2> <a3>, in front of Eiffel tower.
"

TRAIN_ARGS="
    --train_stage="do_ti" \
    --learnable_property="object" \
    --resolution=1024 \
    --num_train_epochs=200 \
    --validation_epochs=100 \
    --num_validation_images=2 \
    --checkpointing_steps=1000000 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --learning_rate=0.0002 \
    --lr_scheduler="constant" --lr_step_rules="1:200,0.4:500,0.1:1000,0.04:2000,0.01" \
    --mixed_precision="fp16" \
    --gradient_checkpointing \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --seed=42 \
"

IFS=', ' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
num_devices=${#devices[@]}

DIST_ARGS="
    --mixed_precision fp16 \
    --num_cpu_threads_per_process 4 \
    --num_processes $num_devices \
    --num_machines 1 \
    --dynamo_backend no \
    --main_process_port $MAIN_PORT \
"
if [ $num_devices -gt 1 ]; then DIST_ARGS+=" --multi_gpu"; fi


cd train

accelerate launch $DIST_ARGS train_text_to_image.py \
    $TRAIN_PATH \
    $TRAIN_ARGS \
    --placeholder_tokens="${placeholder_tokens}" \
    --initializer_tokens="${initializer_tokens}" \
    --validation_prompt="${VALID_PROMPT}"
