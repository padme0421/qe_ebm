python main.py --active_config ML50_en_az_mbart50_config --function test --model mbart --train_size 0 --test_size 0 --val_size 0 \
 --batch_size 16 --dist_strategy deepspeed_stage_2 --adapter --unsup_wt 0.01 --selfsup_strategy beam \
 --score comet_kiwi --from_local_finetuned --checkpoint s3://reinforce-logs/semisupervised_train/mbart/en-az/comet_kiwi/checkpoint/