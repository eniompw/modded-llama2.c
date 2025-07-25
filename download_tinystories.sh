mkdir -p ./modded-llama2.c/data
# Download files in parallel directly into the target directory
wget -P ./modded-llama2.c/data https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok128/tok128.model & \
wget -P ./modded-llama2.c/data https://huggingface.co/datasets/enio/TinyStories/raw/main/tok128/tok128.vocab & \
wget -P ./modded-llama2.c/data https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok128/tok128.tar.gz & \
wait

# Untar the large file after all downloads are complete
tar -I pigz -xf ./modded-llama2.c/data/tok128.tar.gz -C ./modded-llama2.c/data/

# compile run / inference executable
cd modded-llama2.c && gcc -O3 -o run run.c -lm
# create tok105.bin
cd modded-llama2.c && python tokenizer.py --tokenizer-model=data/tok128.model
