#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <cublas_v2.h>

/*
    global - вызывается с CPU, выполняется на GPU
    Функции активации:
    ReLU(x) = max(x, 0)
    Sigmoid(x) = 1/(1 + exp(-x))
*/
__global__ void sigmoid(float* x) 
{
    int idx = threadIdx.x;
    x[idx] = 1 / (1 + exp(-1*x[idx]));
}

__global__ void relu(float* x) 
{
    int idx = threadIdx.x;
    x[idx] = (x[idx]>=0) ? (x[idx]) : 0;
}


cublasHandle_t handler;

/*
    Реализация линейного слоя.
    В инициализации объекта сохраняется размерность слоя.
    В initializer'е 
*/
class FC 
{
public:
    FC() 
    {
        weight = NULL;
        bias = NULL;
        in_features = 0;
        out_features = 0;
    };
    FC(int in, int out) 
    {
        weight = NULL;
        bias = NULL;
        in_features = in;
        out_features = out;
    }

    void init(FILE* weights)
    {
        float* w = (float*)malloc(in_features * out_features * sizeof(float));
        float* b = (float*)malloc(out_features * sizeof(float));

        fread(w, sizeof(float), in_features*out_features, weights);
        fread(b, sizeof(float), out_features, weights);

        cudaMalloc((void**)&weight, in_features * out_features * sizeof(float));
        cudaMalloc((void**)&bias, out_features * sizeof(float));

        cudaMemcpy(weight, w, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias, b, out_features * sizeof(float), cudaMemcpyHostToDevice);
        
        free(w);
        free(b);
    }

    // the vector with the input data is multiplied with the weight matrix
    float* operator() (float* x) 
    {
        const float inc = 1;
        cublasSgemv(handler, CUBLAS_OP_T, in_features, out_features, &inc, weight, in_features, x, 1, &inc, bias, 1);
        cublasScopy(handler, out_features, bias, 1, x, 1);  
        return x;
    }
    ~FC() 
    {
        if (weight)
            cudaFree(weight);
        if (bias)
            cudaFree(bias);
    }
private:
    int in_features;
    int out_features;
    float* weight;
    float* bias;
};

// Нейронна сеть из трёх линейных слоёв
class Net 
{
    FC fc1;
    FC fc2;
    FC fc3;
    // direct dissemination of information
    float forward(float* x) 
    {
        relu<<<1, 256>>>(fc1(x));
        relu<<<1, 16>>>(fc2(x));
        relu<<<1, 1>>>(fc3(x));

        float result;
        cudaMemcpy(&result, x, sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
public:
    Net(int in, int middle1, int middle2) 
    {
        cublasCreate(&handler);
        FILE* weight = fopen("weight.npy", "rb");
        if (weight == NULL) 
        {
            printf(" Error writing in weight file\n");
            exit(1);
        }

        //инициализация переменных размерности
        fc1 = FC(in, middle1);
        fc2 = FC(middle1, middle2);
        fc3 = FC(middle2, 1);

        //Инициализация двумерных массивов - обучаемых параметров weight и bias
        fc1.init(weight);
        fc2.init(weight);
        fc3.init(weight);
    }
    // Launching a neural network. Reading input data from a file
    // and starting a direct flow of information
    float operator() (char* file, int size) 
    {        
        FILE* input = fopen(file, "rb");
        if (input == NULL) 
        {
            printf(" Error writing in input file\n");
            exit(1);
        }
        float* input_layer = (float*)malloc(size * sizeof(float));  
    
        if(input_layer) fread(input_layer, sizeof(float), size, input);

        float* d_layer;
        cudaMalloc((void**)&d_layer, size*sizeof(float));
        cudaMemcpy(d_layer, input_layer, size*sizeof(float), cudaMemcpyHostToDevice);
	    free(input_layer);
        return forward(d_layer);
    }

    ~Net()
    {
        cublasDestroy(handler);
    }
};

int main() 
{
    int size = 1024;
    char input_file[] = "input.npy";
    Net net = Net(1024, 256, 16);    
    float result = net(input_file, size);
    printf("%lf\n\n", result);    
    return 0;
}