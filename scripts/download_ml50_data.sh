# first install fairseq
# set env variables WORKDIR_ROOT=[ML50], SPM_PATH
cd fairseq/examples/multilingual/data_scripts
pip install -r requirement.txt
bash download_ML50_v1.sh
bash preprocess_ML50_v1.sh
# if fairseq not needed, erase