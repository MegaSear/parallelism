/usr/local/cuda/bin/nvcc net.cu -o script_net

./script_net --grid 128
./script_net --grid 256
./script_net --grid 512
./script_net --grid 1024