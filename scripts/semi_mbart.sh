source activate align_reinforce_adapter
#CUDA_VISIBLE_DEVICES=0,1
# sweep over langs
declare -A lang_dict

lang_dict["en_bn"]="ML50_en_bn_mbart50_config"
lang_dict["en_az"]="ML50_en_az_mbart50_config"
lang_dict["en_mr"]="ML50_en_mr_mbart50_config"
lang_dict["en_ka"]="ML50_en_ka_mbart50_config"
lang_dict["en_id"]="ML50_en_id_mbart50_config"
lang_dict["en_pt"]="ML50_en_pt_mbart50_config"
lang_dict["en_ur"]="ML50_en_ur_mbart50_config"

lang_dict["en_kk"]="wmt19_en_kk_mbart50_config"
lang_dict["en_ko"]="iwslt17_en_ko_mbart50_config"
lang_dict["en_lt"]="wmt19_en_lt_mbart50_config"
lang_dict["en_de"]="iwslt17_en_de_mbart50_config"

echo "lang dict done"

# sweep over scores
declare -a scores=("base" "comet_kiwi" "awesome_align" "ensemble")

echo "scores done"

for lang_pair in "${!lang_dict[@]}"; do
    echo $lang_pair
    for score in "${!scores[@]}"; do
        echo ${scores[$score]}
   
        python main.py --active_config ${lang_dict[${lang_pair}]} --function semisupervised_train --model mbart --test_size 0 --val_size 0 \
            --epochs 10 --score ${scores[$score]} --score_list base awesome_align --selfsup_strategy sample --batch_size 16 \
            --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 1 --dir_name $lang_pair --unsup_wt 0.001 \
            --adapter --max_length 50 --gpu_num 1 --min_epoch 1 --weight_schedule \
            --separate_monolingual_dataset --mono_dataset_path bookcorpus --label_train_size 200000 --unlabel_train_size 800000
    done
done