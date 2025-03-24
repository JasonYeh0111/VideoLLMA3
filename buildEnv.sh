pip install -r requirements.txt
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
MAX_JOBS=8 pip install flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
pip install nltk
pip install rouge
pip install bert_score
pip installsentence_transformers
apt update
apt install git-lfs
git lfs install
apt install ffmpeg -y