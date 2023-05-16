pgc++ cpu_code.cpp -o script_cpu_onecore
pgc++ -acc=multicore -Minfo=accel -fast my_net.cpp -o script_cpu_multicore
pgc++ -acc -Minfo=accel -fast my_net.cpp -o script_gpu

##########      GPU      ###########
./script_gpu --grid 128
./script_gpu --grid 256
./script_gpu --grid 512
./script_gpu --grid 1024

########## CPU multicore ###########
./script_cpu_multicore --grid 128
./script_cpu_multicore --grid 256
./script_cpu_multicore --grid 512
./script_cpu_multicore --grid 1024

##########  CPU onecore  ###########
./script_cpu_onecore --grid 128
./script_cpu_onecore --grid 256
./script_cpu_onecore --grid 512
