#include <chrono>
#include <cmath>
#include <malloc.h>
#include <iostream>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

void init(double* net, int size)
{
    net[IDX2C(0, 0, size)] = 10;
    net[IDX2C(0, size - 1, size)] = 20;
    net[IDX2C(size - 1, 0, size)] = 20;
    net[IDX2C(size - 1, size - 1, size)] = 30;

    for (size_t i = 1; i < size - 1; i++)
    {
        net[IDX2C(i, 0, size)] = net[IDX2C(0, 0, size)] + i * (net[IDX2C(size - 1, 0, size)] - net[IDX2C(0, 0, size)]) / (size - 1);
        net[IDX2C(size - 1, i, size)] = net[IDX2C(size - 1, 0, size)] + i * (net[IDX2C(size - 1, size - 1, size)] - net[IDX2C(size - 1, 0, size)]) / (size - 1);
        net[IDX2C(0, i, size)] = net[IDX2C(0, 0, size)] + i * (net[IDX2C(0, size - 1, size)] - net[IDX2C(0, 0, size)]) / (size - 1);
        net[IDX2C(i, size - 1, size)] = net[IDX2C(0, size - 1, size)] + i * (net[IDX2C(size - 1, size - 1, size)] - net[IDX2C(0, size - 1, size)]) / (size - 1);
    }
}

void print(double* net, int size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            printf("%.2f\t", net[IDX2C(i, j, size)]);
        }
        std::cout << std::endl;
    }
}

void algorithm(int size, int epochs, double error_min, bool result)
{
    double* net_new;
    double* net_old;
    net_new = (double*) calloc(size*size, sizeof(double));
    net_old = (double*) calloc(size*size, sizeof(double));
    init(net_new, size);
    init(net_old, size);
    double error = error_min + 1;
    int epoch = 0;
    bool flag;
    for (; epoch < epochs; epoch++)
    {
            double* temp = net_new;
            net_new = net_old;
            net_old = temp;
            flag = !(epoch%size);
            if(flag)
            {
                error = 0;
            }
            for (size_t i = 1; i < size - 1; i++)
            {
                for (size_t j = 1; j < size - 1; j++)
                {
                    net_new[IDX2C(i, j, size)] = 0.25 * (net_old[IDX2C(i + 1, j, size)] + net_old[IDX2C(i - 1, j, size)] +
                                                        net_old[IDX2C(i, j - 1, size)] + net_old[IDX2C(i, j + 1, size)]);
                    if(flag)
                    {       
                        error = std::fmax(error, std::fabs(net_new[IDX2C(i, j, size)] - net_old[IDX2C(i, j, size)]));
                    }
                }
            }
            if(flag)
            {
                if(error < error_min)
                {
                    break;
                }
            }
    }
    std::cout<< "Epoch: " << epoch << std::endl;
    std::cout << "Error: " << error << std::endl;
    if (result)
    {
        std::cout << "Calculated matrix:" << std::endl;
        print(net_new, size);
    }
    free(net_new);
    free(net_old);
}

int main(int argc, char* argv[])
{
    double accuracy = 1e-6;
    int max_iteration = (int)1e6;
    int length_grid = 1024;
    bool result = false;

    for (int arg = 1; arg < argc; arg++)
    {
        std::string str = argv[arg];
        if (!str.compare("--res"))
            result = true;
        else if (!str.compare("--acc"))
            accuracy = std::stod(argv[++arg]);
        else if (!str.compare("--iter"))
            max_iteration = (int)std::stod(argv[++arg]);
        else if (!str.compare("--grid"))
            length_grid = std::stoi(argv[++arg]);
        else
        {
            std::cout << "Arg unexpected!";
            return -1;
        }
    }
    std::cout << "--------------------------------" <<std::endl;
    std::cout << "---PROGRAMM START---" << std::endl;
    std::cout << "Current options:" << std::endl;
    std::cout << "\tSize network: " << length_grid << std::endl;
    std::cout << "\tMax Iterations: " << max_iteration << std::endl;
    std::cout << "\tAccuracy: " << accuracy << std::endl;

    auto begin_main = std::chrono::steady_clock::now();
    algorithm(length_grid, max_iteration, accuracy, result);
    auto end_main = std::chrono::steady_clock::now();
    
    int time_spent_main = std::chrono::duration_cast<std::chrono::milliseconds>(end_main - begin_main).count();
    std::cout << "Elapsed time: " << time_spent_main << " ms" << std::endl;
    std::cout << "---PROGRAMM FINISH---" <<std::endl;
    std::cout << "--------------------------------" <<std::endl << std::endl;
    return 0;
}
