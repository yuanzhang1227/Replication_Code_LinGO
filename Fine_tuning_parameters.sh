# Fine-tuning is operated with Llamafactory
# We experimented with Mistral-7B, DeepSeek--V2--Lite--Chat, Qwen/Qwen3-4B-Instruct-2507

!llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset lora_training_incivility_human \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir human_qwen3_instruct \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --warmup_ratio 0.1 \
    --save_steps 1000 \
    --learning_rate 8e-5 \
    --num_train_epochs 3.0 \
    --max_samples 500 \
    --max_grad_norm 1.0 \
    --loraplus_lr_ratio 16.0 \
    --fp16 \
    --report_to none