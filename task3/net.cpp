#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>
#include <cublas_v2.h>

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \

/*
define для удобного обращения к одномерному массиву, нужно для более краткой записи кода.
*/
#define IDX2C(i, j, ld) (((j)*(ld))+(i))


/*
Функция инициализации матриц
*/
void init(double* net, double num, int size)
{
    /*
    Заполнение матрциы некоторым числом (например нулём, но в лучшем случае 
    код будет испольняться быстрее, если заполнить средним арифметическим чисел углов сетки).
    Двойной цикл является независимым по итерациям.
    */
    #pragma acc kernels deviceptr(net)
    memset(net, num, size*size);
    /*
    Заполнение угловых значений матрицы
    */
    #pragma acc parallel deviceptr(net)
    {
    net[IDX2C(0, 0, size)] = 10;
    net[IDX2C(0, size - 1, size)] = 20;
    net[IDX2C(size - 1, 0, size)] = 20;
    net[IDX2C(size - 1, size - 1, size)] = 30;
    }
    /*
    Заполнение боковых значений матрицы. Расчёт значений производится таким образом, чтобы итерации были независмыми
    */
    #pragma acc parallel loop independent deviceptr(net)
    for (size_t i = 1; i < size - 1; i++)
    {
        net[IDX2C(i, 0, size)] = net[IDX2C(0, 0, size)] + i * (net[IDX2C(size - 1, 0, size)] - net[IDX2C(0, 0, size)]) / (size - 1);
        net[IDX2C(size - 1, i, size)] = net[IDX2C(size - 1, 0, size)] + i * (net[IDX2C(size - 1, size - 1, size)] - net[IDX2C(size - 1, 0, size)]) / (size - 1);
        net[IDX2C(0, i, size)] = net[IDX2C(0, 0, size)] + i * (net[IDX2C(0, size - 1, size)] - net[IDX2C(0, 0, size)]) / (size - 1);
        net[IDX2C(i, size - 1, size)] = net[IDX2C(0, size - 1, size)] + i * (net[IDX2C(size - 1, size - 1, size)] - net[IDX2C(0, size - 1, size)]) / (size - 1);
    }
}


/*
Функция вывода матрицы (находящейся в памяти ускорителя), для удобства.
*/
void print(double* net, int size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            #pragma acc kernels deviceptr(net)
            printf("%.2f\t", net[IDX2C(i, j, size)]);
        }
        std::cout << std::endl;
    }
}


/*
Основная функция для подсчёта матрицы. 
Здесь просиходит: 1. Инициализация матриц. 
                  2. Подсчёт более приближённой к ответу матрицы.
                  3. Расчёт ошибки из старой матрицы и новой, ошибка уменьшается в процессе, так как алгоритм сходится.
Алгоритм работает, пока ошибка больше той, которая необходима. Также пристутсвует ограничение по количеству итераций.
*/
void algorithm(int size, int epochs, double error_min, bool result)
{
    double num = 0;
    double* net_new;
    double* net_old;
    net_new = (double*) acc_malloc(size*size*sizeof(double));
    net_old = (double*) acc_malloc(size*size*sizeof(double));

    const double alpha = -1;
    const int inc = 1;
    int max_idx = 0;

    init(net_new, num, size);
    init(net_old, num, size);

    double error = error_min + 1;
    int epoch  = 0;
    bool flag;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    #pragma acc data copy(error)
    {
        for (; epoch < epochs; epoch++)
        {
            /*
            Подсчёт ошибки происходит не каждую итерацию, а один раз за количество итераций равному размеру матрицы.
            size больше => Считаем ошибку реже.
            */
            flag = !(epoch%size);
            #pragma acc parallel loop independent collapse(2) deviceptr(net_old, net_new)
            for (size_t j = 1; j < size - 1; j++)
            {
                for (size_t i = 1; i < size - 1; i++)
                {
                    /*
                    Подсчёт новой матрицы на основе старой по пятиточечному шаблону.
                    */
                    net_new[IDX2C(i, j, size)] = 0.25 * (net_old[IDX2C(i + 1, j, size)] + net_old[IDX2C(i - 1, j, size)] +
                                                        net_old[IDX2C(i, j - 1, size)] + net_old[IDX2C(i, j + 1, size)]);
                }
            }
            if(flag)
            {
                #pragma acc kernels
                error = 0;
                #pragma acc data deviceptr(net_new, net_old)
                {
                    CUBLAS_CHECK(cublasDaxpy(handle, size*size, &alpha, net_new, inc, net_old, inc));
                }
                CUBLAS_CHECK(cublasIdamax(handle, size*size, net_old, inc, &max_idx));
                #pragma acc data update deviceptr(net_new, net_old)
                {
                    #pragma acc kernels
                    error = fabs(net_old[max_idx - 1]);
                }
                /*
                Одно из условий для завершения расчёта матрицы
                */
                #pragma acc update host(error)
                if(error <= error_min)
                {
                    break;
                }
                #pragma acc data deviceptr(net_new, net_old)
                {
                CUBLAS_CHECK(cublasDcopy(handle, size*size, net_new, inc, net_old, inc));
                }
            }
            else
            { 
                /*
                Swap указателей быстрее, чем swap значений матрицы.
                */
                double* temp = net_new;
                net_new = net_old;
                net_old = temp;
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

    acc_free(net_new);
    acc_free(net_old);
}

int get_free_device()
{
    FILE *fp;
    int MAX_LINE_LENGTH = 1024;
    char buffer[MAX_LINE_LENGTH];
    float min_usage = 100.0;
    int device = 0;
    fp = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv", "r");

    if (fp == NULL) 
    {
        std::cout << "Failed to execute command" << std::endl;
        exit(1);
    }

    std::cout << "Device list and using:" << std::endl;
    for(int i = 0; fgets(buffer, MAX_LINE_LENGTH, fp) != NULL; i++)
    {
        if (i < 1)
        {
            continue;
        }
        int usage = atoi(strtok(buffer, "%"));
        std::cout << "\tDevice: " << i-1 << "\tUsing: " << usage << "%" << std::endl;

        if (usage <= min_usage) 
        {
            min_usage = int(usage);
            device = i-1;
        }
    }

    std::cout<< "Free device: " << device << std::endl;
    std::cout<< "Minimum usage: " << min_usage << "%" << std::endl;
    return device;
}
/*
Main функция, здесь происходит: 1. Подсчёт времени выполнения алгоритма
                                2. Определяется ускоритель 
                                3. Парсер аргументов
                                3. Вызывается функция для расчёта матрицы
                                4. Красивый вывод параметров
*/
int main(int argc, char* argv[])
{
    /*
    Выбор либо свободного ускорителя.
    */
    
    int device = get_free_device();
    acc_set_device_num(device, acc_device_default);
    

    /*
    Объявление основых констант, таких как:
    размер матрицы, точность алгоритма а также макисмально допустимого количества итераций
    */
    double accuracy = 1e-6;
    int max_iteration = (int)1e6;
    int length_grid = 1024;
    bool result = false;

    /*
    Парсер аргументов
    */
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

    /*
    Вызов функции подсчёта матрицы а также подсчёт времени этого алгоритма.
    */
    auto begin_main = std::chrono::steady_clock::now();
    algorithm(length_grid, max_iteration, accuracy, result);
    auto end_main = std::chrono::steady_clock::now();

    int time_spent_main = std::chrono::duration_cast<std::chrono::milliseconds>(end_main - begin_main).count();
    std::cout << "Elapsed time: " << time_spent_main << " ms" << std::endl;
    std::cout << "---PROGRAMM FINISH---" <<std::endl;
    std::cout << "--------------------------------" <<std::endl << std::endl;
    return 0;
}
