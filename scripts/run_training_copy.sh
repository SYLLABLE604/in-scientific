deepspeed_path=$(which deepspeed)
if [ -z "$deepspeed_path" ]; then
    echo "deepspeed not found. Please make sure it is installed and added to your PATH."
    exit 1
fi
#需要修改deepspeed，model_name_or_path，output_dir，data_type，data_path,output_dir,per_device_train_batch_size,model_max_length,
# dataloader_num_workers,mmc4_max_num_images

# 不用管的mm_projector_type，mm_vision_select_layer，num_train_epochs，vision_tower，tune_mm_mlp_adapter，mm_projector_type，image_aspect_ratio
# mm_use_im_start_end，mm_use_im_patch_token,bf16,num_train_epochs,per_device_eval_batch_size,gradient_accumulation_steps,evaluation_strategy
# save_strategy,save_steps,save_total_limit, learning_rate, weight_decay,lr_scheduler_type,logging_steps,tf32,lazy_preprocess,freeze_mm_mlp_adapter

# 不知道改不改的，image_folder, gradient_checkpointing
$deepspeed_path --include="localhost:1,2,3,4,5,6,7" --master_port=61000 /root/model/multimodal_textbook/llava/train/train_interleaved_mem.py \
    --deepspeed /root/model/multimodal_textbook/scripts/zero2.json \
    --model_name_or_path /root/model/llava-v1.5-7b \
    --version llava_v1 \
    --data_type "inscientific" \
    --data_path /root/model/in_dataset/data/IN-Arxiv/part01/part01.jsonl \
    --image_folder /root/model/in_dataset/data/IN-Arxiv/part01/content_image \
    --vision_tower /root/model/clip \
    --tune_mm_mlp_adapter False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --num_train_epochs 1 \
    --output_dir /root/model/multimodal_textbook/output\
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --freeze_mm_mlp_adapter True \
    --h100 False \
    --max_num_images 12
