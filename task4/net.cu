#include <chrono>
#include <cmath>
#include <iostream>
#include <cstring>
#include <chrono>
#include "cuda_runtime.h"
#include <cub/cub.cuh>

/* 
    проверка cuda error
*/
#define CUDA_CHECK(err)                                                        \
    {                                                                          \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess)                                               \
        {                                                                      \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    }                                                                          \

/*
    define для удобного обращения к одномерному массиву, нужно для более краткой записи кода.
*/
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

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

// Посчитать матрицу
__global__ void calculate_matrix(double *Anew, double *A, uint32_t size)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    // Граница или выход за границы массива - ничего не делать
    if (i >= size - 1 || j >= size - 1 || i <= 0 || j <= 0)
        return;

    Anew[IDX2C(i, j, size)] = (A[IDX2C(i + 1, j, size)] + A[IDX2C(i - 1, j, size)] + A[IDX2C(i, j - 1, size)] + A[IDX2C(i, j + 1, size)]) * 0.25;
}

// O = |A-B|
__global__ void count_matrix_difference(double *matrixA, double*matrixB, double*outputMatrix, uint32_t size)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    // Выход за границы массива или периметр - ничего не делать
    if (i >= size - 1 || j >= size - 1 || i == 0 || j == 0)
        return;

    uint32_t idx = IDX2C(i, j, size);
    outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
}

void algorithm(const int net_size = 128, const int iter_max = 1e6, const double accuracy = 1e-6, const bool res = false)
{
    const size_t vec_size = net_size * net_size;

    uint32_t threads_in_block = 32;                      // Потоков в одном блоке (32 * 32 максимум)
    uint32_t block_in_grid = ceil((double)net_size / threads_in_block); // Блоков в сетке (size / 32 максимум)

    dim3 blockPerGrid = dim3(block_in_grid, block_in_grid);
    dim3 threadPerBlock = dim3(threads_in_block,threads_in_block);

    // Матрица на хосте (нужна только для инициализации и вывода)
    double*A;

    CUDA_CHECK(cudaMallocHost(&A, sizeof(double) * vec_size));
    double num = 0;
    // Инициализация матрицы
    init(A, num, net_size);

    // Создание 2-х матриц на девайсе
    double*A_dev, *Anew_dev;
    CUDA_CHECK(cudaMalloc(&A_dev, sizeof(double) * vec_size));
    CUDA_CHECK(cudaMalloc(&Anew_dev, sizeof(double) * vec_size));

    // Поток для копирования
    cudaStream_t memory_stream;
    CUDA_CHECK(cudaStreamCreate(&memory_stream));

    // Скопировать матрицу с хоста на матрицы на девайсе
    CUDA_CHECK(cudaMemcpy(A_dev, A, sizeof(double) * vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Anew_dev, A, sizeof(double) * vec_size, cudaMemcpyHostToDevice));

    // Текущая ошибка
    double*error, *error_dev;
    CUDA_CHECK(cudaMallocHost(&error, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&error_dev, sizeof(double)));
    *error = accuracy + 1;

    // Матрица ошибок
    double*A_err;
    CUDA_CHECK(cudaMalloc(&A_err, sizeof(double) * vec_size));

    // Временный буфер для редукции и его размер
    double*reduction_bufer = NULL;
    uint64_t reduction_bufer_size = 0;

    // Первый вызов, чтобы предоставить количество байтов, необходимое для временного хранения, необходимого CUB.
    cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size);

    // Выделение памяти под буфер
    CUDA_CHECK(cudaMalloc(&reduction_bufer, reduction_bufer_size));

    // Граф
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;

    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Сокращение обращений к CPU. Больше сетка - реже стоит сверять значения.
    uint32_t num_skipped_checks = (iter_max < net_size) ? iter_max : net_size;
    num_skipped_checks += num_skipped_checks % 2; // Привести к четному числу

    for (uint32_t k = 0; k < num_skipped_checks; k += 2)
    {
        calculate_matrix<<<blockPerGrid, threadPerBlock, 0, stream>>>(A_dev, Anew_dev, net_size);
        calculate_matrix<<<blockPerGrid, threadPerBlock, 0, stream>>>(Anew_dev, A_dev, net_size);
    }

    count_matrix_difference<<<blockPerGrid, threadPerBlock, 0, stream>>>(A_dev, Anew_dev, A_err, net_size);
   
    // Найти максимум и положить в error_dev - аналог reduction (max : error_dev) в OpenACC
    cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size, stream);

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Счетчик итераций
    int iter = 0;

    // Вывод
    // if (res)
    //    print(A, net_size);

    // Начать отсчет времени работы
    auto begin_main = std::chrono::steady_clock::now();
    for (; iter < iter_max && *error > accuracy; iter += num_skipped_checks)
    {
        // Запуск графа
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));

        // Синхронизация потока
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Копировать ошибку с девайса на хост
        CUDA_CHECK(cudaMemcpy(error, error_dev, sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::cout<< "Epoch: " << iter << std::endl;
    std::cout << "Error: " << *error << std::endl;
    // Вывод
    //if (res)
    //{
    //    CUDA_CHECK(cudaMemcpyAsync(A, A_dev, sizeof(double) * vec_size, cudaMemcpyDeviceToHost, memory_stream));
    //    print(A, net_size);
    //}

    // Освобождение памяти
    CUDA_CHECK(cudaFree(reduction_bufer));
    CUDA_CHECK(cudaFree(A_err));
    CUDA_CHECK(cudaFree(A_dev));
    CUDA_CHECK(cudaFree(Anew_dev));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(error));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(memory_stream));
    CUDA_CHECK(cudaGraphDestroy(graph));
}



/*
    Функция поиска свободного устройства. Получает результат команды nvidia-smi и выбирает менее загруженный ускоритель
*/
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

    std::cout << "List of devices and using:" << std::endl;
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

    std::cout<< "\nSelected device:" << "\tDevice: " << device << "\tMinimum Using: " << min_usage << "%" << std::endl << std::endl;
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

    /*
        Выбор либо свободного ускорителя.
    */
    
    int device = get_free_device();
    CUDA_CHECK(cudaSetDevice(device));
    
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



// Вывести свойства девайса
void print_device_properties(void)
{
    cudaDeviceProp deviceProp;
    if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, 0))
    {
        printf("Warp size in threads is %d.\n", deviceProp.warpSize);
        printf("Maximum size of each dimension of a block is %d, %d, %d.\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid is %d, %d, %d.\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum resident threads per multiprocessor is %d.\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of resident blocks per multiprocessor is %d.\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("_____________________________________________________________________________________________\n");
    }
}

