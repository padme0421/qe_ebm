#conda create -n align_reinforce_adapter python=3.10 libffi=3.3
#source ~/anaconda/etc/profile.d/conda.sh
#conda activate align_reinforce_adapter

env_name="align_reinforce_adapter"

# Check if the Conda environment exists
if conda info --envs | grep -q "^$env_name "; then
    echo "Conda environment '$env_name' exists."
else
    echo "Conda environment '$env_name' does not exist."
    exit 1
fi

# Get the currently activated Conda environment
current_env=$(echo $CONDA_DEFAULT_ENV)

# Check if the specified environment is activated
if [ "$current_env" == "$env_name" ]; then
    echo "The '$env_name' environment is currently activated."
else
    echo "The '$env_name' environment is not activated in the current shell."
    exit 1
fi

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# (for wsl) pip3 install torch torchvision torchaudio
#conda install -c pytorch
conda install -c conda-forge portalocker
conda install lightning==2.0.8 -c conda-forge
pip install spacy
pip install matplotlib
python -m spacy download de_core_news_md
python -m spacy download en_core_web_md
python -m spacy download ko_core_news_md
pip install pandas
pip install nltk
pip install transformers
pip install adapters
pip install datasets
pip install evaluate
pip install sacremoses
pip install "sacrebleu[ko]"
pip install sentencepiece
pip install revtok
pip install konlpy
pip install wandb
pip install protobuf==3.20.3
pip install -U deepspeed
pip install undecorated
pip install openai==0.27.8
pip install python-dotenv==1.0.0
pip install s3fs==0.4.2
#pip install fairseq
pip install unbabel-comet
pip install "numpy<1.24,>= 1.22.4"
pip install peft
pip install pylint
pip install info-nce-pytorch
pip install "unbabel-comet>=2.2.0"
pip install jsonlines
# install trl from source (git clone)

# install fairseq editable to use cmlm

pip install sentence-transformers
