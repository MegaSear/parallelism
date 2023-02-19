#include <chrono>
#include <iostream>
#include <cmath>
double PI = 3.141592653589793238462643383279;
#define time_s std::chrono::steady_clock::now

float calc_sum(float *arr, int size)
{
    double sum = 0;
#pragma acc data copy(sum)
    {
#pragma acc parallel loop reduction(+:sum)
            for (size_t i = 0; i < size; i++)
            {
                sum += arr[i];
            }
    }
    return sum;
}
void calc_array(float *arr, int size)
{
    double step = 2 * PI / size;
#pragma acc data copyin(step)
    {
#pragma acc parallel loop present(arr[:size])
        for (size_t i = 0; i < size; i++)
        {
            arr[i] = sin((double)(i * step));
        }
    }
}

int main()
{
    double counter_sum =0, counter_tmain= 0, counter_tarray =0, counter_tsum = 0;

    for(int i = 0; i < 100; i++) //считаем среднее
    {

    auto start_main = time_s();
    int size = 1e7;
    float *arr = new float[size];

#pragma acc enter data create(arr[:size])

    auto start_array = time_s();
    calc_array(arr, size);
    auto end_array = time_s();

    auto start_sum = time_s();
    float sum = calc_sum(arr, size);
    auto end_sum = time_s();

#pragma acc exit data delete(arr[:size])

    delete[] arr;
    auto end_main = time_s();

    auto time_main = std::chrono::duration_cast<std::chrono::microseconds>(end_main - start_main).count();
    auto time_array = std::chrono::duration_cast<std::chrono::microseconds>(end_array - start_array).count();
    auto time_sum = std::chrono::duration_cast<std::chrono::microseconds>(end_sum - start_sum).count();


    counter_tsum += time_sum;
    counter_tmain += time_main;
    counter_tarray += time_array;
    counter_sum += sum;
    }
    
    double avsum = counter_sum/100.0;
    double avtsum = counter_tsum/100.0;
    double avtmain = counter_tmain/100.0;
    double avtarray = counter_tarray/100.0;
    std::cout << "Calculations:\n"
              << "\tSum = " << avsum;
    std::cout << "\nThe elapsed time: \n"
            << "\tMain " << avtmain << " us\n"
            << "\tArray " << avtarray << " us\n"
            << "\tSum " << avtsum << " us\n";

    return 0;
}