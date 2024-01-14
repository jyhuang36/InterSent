python run.py \
--data_dir data \
--do_test \
--tasks add diff extcomp \
--pooler_type cls \
--output output \
--n_gpus 1 \
--batch_size_per_gpu 64 \
--checkpoint output/lightning_logs/version_0/checkpoints/checkpoint.ckpt \