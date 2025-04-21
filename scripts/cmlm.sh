#CUDA_VISIBLE_DEVICES=0,1
# sweep over langs
declare -A lang_dict
#lang_dict["en_gu"]="wmt19_en_gu_mbart50_config" # low resource
#lang_dict["en_kk"]="wmt19_en_kk_mbart50_config" # low resource
#lang_dict["en_ko"]="iwslt17_en_ko_mbart50_config" # medium resource, distant lang
#lang_dict["en_ko"]="iwslt17_en_ko_xlmr_config"
#lang_dict["en_lt"]="wmt19_en_lt_mbart50_config" # medium resource, simlar (?) lang
#lang_dict["en_zh"]="wmt19_en_zh_mbart50_config" # high resource, distant lang
#lang_dict["en_de"]="wmt19_en_de_mbart50_config" # high resource, similar lang
lang_dict["en_de"]="iwslt17_en_de_mbart50_config"
#lang_dict["en_bn"]="ML50_en_bn_mbart50_config"
#lang_dict["en_az"]="ML50_en_az_mbart50_config"
#lang_dict["en_mr"]="ML50_en_mr_mbart50_config"
#lang_dict["en_mn"]="ML50_en_mn_mbart50_config"

#lang_dict["en_sv"]="ML50_en_sv_mbart50_config"
#lang_dict["en_th"]="ML50_en_th_mbart50_config"
#lang_dict["en_id"]="ML50_en_id_mbart50_config"
#lang_dict["en_pt"]="ML50_en_pt_mbart50_config"
#lang_dict["en_ur"]="ML50_en_ur_mbart50_config"
#lang_dict["en_ur"]="ML50_en_ur_xlm_config"
#lang_dict["en_mk"]="ML50_en_mk_mbart50_config"
#lang_dict["en_te"]="ML50_en_te_mbart50_config"
#lang_dict["en_sl"]="ML50_en_sl_mbart50_config"
#lang_dict["en_ka"]="ML50_en_ka_mbart50_config"
#lang_dict["en_ka"]="ML50_en_ka_xlm_config"
#lang_dict["en_ro"]="wmt16_en_ro_mbart50_config"

echo "lang dict done"

# sweep over scores
declare -a scores=("base") #("awesome_align") #("comet_kiwi" "awesome_align" "dep_parse_base_align")

echo "scores done"

for lang_pair in "${!lang_dict[@]}"; do
    echo $lang_pair
    for score in "${!scores[@]}"; do
        echo ${scores[$score]}
        #python main.py --active_config ${lang_dict[${lang_pair}]} --function supervised_train --model nambart --train_size 200000 --test_size 0 --val_size 0 \
        #    --epochs 10 --score ${scores[$score]} --selfsup_strategy sample --batch_size 8 \
        #    --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 1 --dir_name $lang_pair \
        #    --adapter --max_length 50

        #checkpoint_url=$(<checkpoint_url.txt) 

        #python main.py --active_config ${lang_dict[${lang_pair}]} --function test --model mbart --train_size 200000 --test_size 0 --val_size 0 \
        #    --batch_size 16 --dist_strategy deepspeed_stage_2 --adapter --unsup_wt 0.001 --selfsup_strategy beam \
        #    --score ${scores[$score]} --from_local_finetuned --checkpoint $checkpoint_url --max_length 50
        
        #rm checkpoint_url.txt
   
        python main.py --active_config ${lang_dict[${lang_pair}]} --function semisupervised_train --model cmlm --train_size 200000 --test_size 0 --val_size 0 \
            --epochs 10 --score ${scores[$score]} --selfsup_strategy sample --batch_size 8 \
            --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 2 --dir_name $lang_pair --unsup_wt 0.001 \
            --adapter --max_length 50 --gpu_num 1

        #checkpoint_url=$(<checkpoint_url.txt) 

        #python main.py --active_config ${lang_dict[${lang_pair}]} --function test --model mbart --train_size 200000 --test_size 0 --val_size 0 \
        #   --batch_size 16 --dist_strategy deepspeed_stage_2 --adapter --unsup_wt 0.001 --selfsup_strategy beam \
        #    --score ${scores[$score]} --from_local_finetuned --checkpoint $checkpoint_url --max_length 50
        
        #rm checkpoint_url.txt

    done
done