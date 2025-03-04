DATA_PATH=[Your DiDeMo data and videos path]
CKPT_PATH=[Directory to save model weights and checkpoints during training]

python -m torch.distributed.launch --nproc_per_node=1 \
main_narvid.py \
--do_train --num_thread_reader=4 --epochs=15 --batch_size=32 --n_display=20 \
--data_path ${DATA_PATH}/didemo_data \
--narration_path ${DATA_PATH}/didemo_data/didemo_narration.json \
--features_path ${DATA_PATH}/frames \
--output_dir ${CKPT_PATH} \
--lr 1e-4 --max_words 64 --max_frames 32 --batch_size_val 32 \
--datatype didemo \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--hard_negative_loss_factor 1.9 --hard_negative_selection_factor 1.0 \
--hard_negative_weighting 1 --nucleus_P 0.5 \
--pretrained_clip_name ViT-B/16