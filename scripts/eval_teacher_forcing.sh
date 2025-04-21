# sweep over langs
declare -A lang_dict
lang_dict["en_gu"]="wmt19_en_gu_mbart50_config" # low resource
lang_dict["en_kk"]="wmt19_en_kk_mbart50_config" # low resource
lang_dict["en_ko"]="iwslt17_en_ko_mbart50_config" # medium resource, distant lang
lang_dict["en_lt"]="wmt19_en_lt_mbart50_config" # medium resource, simlar (?) lang
#lang_dict["en_zh"]="wmt19_en_zh_mbart50_config" # high resource, distant lang
#lang_dict["en_de"]="wmt19_en_de_mbart50_config" # high resource, similar lang

echo "lang dict done"

# sweep over scores
declare -a scores=("base" "fast_align") #("comet" "awesome_align" "dep_parse")

echo "scores done"

for lang_pair in "${!lang_dict[@]}"; do
    echo $lang_pair
    for score in "${!scores[@]}"; do
        echo ${scores[$score]}
        #python main.py --active_config ${lang_dict[${lang_pair}]} --function supervised_train --model mbart --train_size 200000 --test_size 0 --val_size 0 \
        #    --epochs 1 --score ${scores[$score]} --selfsup_strategy beam_sample --batch_size 16 \
        #    --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 1 --dir_name $lang_pair \
        #    --adapter --max_length 50 --min_epoch 0

        checkpoint_url=$(<checkpoint_url.txt) 

        python main.py --active_config ${lang_dict[${lang_pair}]} --function test --model mbart --train_size 200000 --test_size 0 --val_size 0 \
            --batch_size 16 --dist_strategy deepspeed_stage_2 --adapter --unsup_wt 0.001 --selfsup_strategy beam \
            --score ${scores[$score]} --from_local_finetuned --checkpoint $checkpoint_url --max_length 50 --min_epoch 0
        
        rm checkpoint_url.txt
   
        #python main.py --active_config ${lang_dict[${lang_pair}]} --function semisupervised_train --model mbart --train_size 200000 --test_size 0 --val_size 0 \
        #    --epochs 1 --score ${scores[$score]} --selfsup_strategy beam_sample --batch_size 16 \
        #    --dist_strategy deepspeed_stage_2 --accumulate_grad_batches 1 --dir_name $lang_pair --unsup_wt 0.001 \
        #    --adapter --max_length 50 --min_epoch 0

        #checkpoint_url=$(<checkpoint_url.txt) 

        #python main.py --active_config ${lang_dict[${lang_pair}]} --function test --model mbart --train_size 200000 --test_size 0 --val_size 0 \
        #    --batch_size 16 --dist_strategy deepspeed_stage_2 --adapter --unsup_wt 0.001 --selfsup_strategy beam \
        #   --score ${scores[$score]} --from_local_finetuned --checkpoint $checkpoint_url --max_length 50 --min_epoch 0
        
        #rm checkpoint_url.txt

    done
done