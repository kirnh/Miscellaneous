import os

for fold_number in ['0']:
    # Command line script for training
    train_script = 'python train_image_classifier.py \
    --train_dir ./new_checkpoints/inception_v3/fold_%s \
    --dataset_dir ./patches_wo_background/exp_2/fold_%s \
    --dataset_name cells --dataset_split train \
    --model_name inception_v3 \
    --checkpoint_path ./checkpoints/inception_v3.ckpt \
    --checkpoint_exclude_scopes InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes InceptionV3/Logits,InceptionV3/AuxLogits \
    --max_number_of_steps 50 \
    --save_summaries_secs 1' %(fold_number, fold_number)
    # Execute the training script
    os.system(train_script)
    # Command line script for evaluating
    eval_script = 'python eval_image_classifier.py \
    --checkpoint_path ./new_checkpoints/inception_v3/fold_%s \
    --dataset_dir ./patches_wo_background/exp_2/fold_%s \
    --dataset_name cells \
    --dataset_split_name validation \
    --model_name inception_v3 \
    --eval_dir ./eval_dir/fold_%s' %(fold_number, fold_number, fold_number)
    # Execute the evaluation script
    os.system(eval_script)

