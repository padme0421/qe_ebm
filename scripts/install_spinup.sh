# install openmpi
sudo apt-get update && sudo apt-get install libopenmpi-dev

# install spinning up
# (first move to home directory)
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .