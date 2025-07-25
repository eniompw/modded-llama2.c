mkdir -p ./modded-llama2.c/data/tok128
cd ./modded-llama2.c/data
# Download files in parallel directly into the target directory
wget https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok128/tok128.model & \
wget https://huggingface.co/datasets/enio/TinyStories/raw/main/tok128/tok128.vocab & \
wget https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok128/tok128-01.tar.gz & \
wait

# Untar the large file after all downloads are complete
tar -I pigz -xvf tok128-01.tar.gz -C ./tok128/

# compile run / inference executable
cd .. && gcc -O3 -o run run.c -lm
# create tok105.bin
python tokenizer.py --tokenizer-model=data/tok128.model
