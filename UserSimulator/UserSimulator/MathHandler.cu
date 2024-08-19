#include "MathHandler.cuh"

//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <cfloat>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while(0)

MathHandler::MathHandler(unsigned long randomSeed) : _randSeed(randomSeed), _re(randomSeed), _blockSize(16,16)  {

}

enum MatrixOperation {Add, AddInvert, Substract, SubstractInvert, Multiply, Divide, SquaredSubstractInvert, DivideExp, Exp, OneHotEncodedVector, ReLUDerivative, LeakyReLUDerivative, SigmoidDerivative, None};

#pragma region CUDA kernels
// Kernel to add a column vector to each column of a matrix
__global__ void addColumnVectorToMatrix(double* matrix, double* vector, double* result, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the matrix
    if (row < m && col < n) {
        result[row * n + col] = matrix[row * n + col] + vector[row];
    }
}

// computre row averages of matrix MxN and return row vector Mx1
__global__ void rowAverage(const double* matrix, double* rowAvg, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double sum = 0.0;
        for (int col = 0; col < N; ++col) {
            sum += matrix[row * N + col];
        }
        rowAvg[row] = sum / N;
    }
}

// CUDA kernel to apply tanh to each element of the matrix
__global__ void actFuncKernel(double* d_matrix, int width, int height, LayerActivationFuncs fn) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        if (fn == tanhAct) d_matrix[idx] = tanh(d_matrix[idx]);
        else if (fn == reluAct) d_matrix[idx] = fmax(0.0, d_matrix[idx]);
        else if (fn == leakyReLU) d_matrix[idx] = d_matrix[idx] > 0 ? d_matrix[idx] : 0.01 * d_matrix[idx];
        else if (fn == sigAct) d_matrix[idx] = 1 / (1 + exp(-d_matrix[idx]));
    }
}

// CUDA kernel to transpose a matrix
__global__ void transposeKernel(double* d_out, const double* d_in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx_in = y * width + x;
        int idx_out = x * height + y;
        d_out[idx_out] = d_in[idx_in];
    }
}

// Kernel function to perform matrix multiplication
__global__ void matrixMultiplyKernel(double* A, double* B, double* C, int numARows, int numAColumns, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        double value = 0.0;
        for (int k = 0; k < numAColumns; ++k) {
            value += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numBColumns + col] = value;
    }
}

__global__ void matrixElementWiseKernel(double* A, double* B, double* C, int numRows, int numCols, MatrixOperation op) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        if (op == Substract) C[index] = A[index] - B[index];
        else if (op == Add) C[index] = A[index] + B[index];
        else if (op == Multiply) C[index] = A[index] * B[index];
        else if (op == Divide) C[index] = A[index] / B[index];
    }
}

__global__ void matrixElementWiseKernel(double* A, double constValue, double* C, int numRows, int numCols, MatrixOperation op) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        if (op == Substract) C[index] = A[index] - constValue;
        else if (op == Divide) C[index] = A[index] / constValue;
        else if (op == DivideExp) C[index] = std::exp(A[index]) / constValue;
        else if (op == Add) C[index] = A[index] + constValue;
        else if (op == Multiply) C[index] = A[index] * constValue;
        else if (op == SubstractInvert) C[index] = -(A[index] - constValue);
        else if (op == SquaredSubstractInvert) C[index] = -(pow(A[index], 2) - constValue);
        else if (op == Exp) C[index] = std::exp(A[index]);
        else if (op == ReLUDerivative) C[index] = A[index] > 0.0 ? 1.0 : 0.0;
        else if (op == LeakyReLUDerivative) C[index] = A[index] > 0.0 ? 1.0 : 0.01;
        else if (op == SigmoidDerivative) {
            double sigmoid = 1.0 / (1.0 + exp(-A[index]));
            C[index] = sigmoid * (1.0 - sigmoid);
        }
    }
}

// kernel for vector addition
__global__ void addVecKernel(double* c, const double* a, const double* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
#pragma endregion

#pragma region kernel calls
void AddMatrixAndColumnVec(double* matrix, double* vector, double* res, int m, int n) {

    // Device matrices
    double* d_matrix, * d_vector, * d_result;

    // Allocate device memory
    cudaMalloc(&d_matrix, m * n * sizeof(double));
    cudaMalloc(&d_vector, m * sizeof(double));
    cudaMalloc(&d_result, m * n * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_matrix, matrix, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, m * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    addColumnVectorToMatrix << <gridSize, blockSize >> > (d_matrix, d_vector, d_result, m, n);

    // Copy result from device to host
    cudaMemcpy(res, d_result, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}

CUDAMatrix RowVectorFromRowAverage(CUDAMatrix* mat) {
    //double h_rowAvg[M];

    int rows = mat->GetRows();
    int cols = mat->GetColumns();

    double* h_rowAvg = new double[rows];

    // Allocate device memory
    double* d_matrix, * d_rowAvg;
    cudaMalloc(&d_matrix, rows * cols * sizeof(double));
    cudaMalloc(&d_rowAvg, rows * sizeof(double));

    // Copy the host matrix to the device
    cudaMemcpy(d_matrix, mat->GetUnderlyingMatrix(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;

    // Launch the kernel to compute row averages
    rowAverage << <gridSize, blockSize >> > (d_matrix, d_rowAvg, rows, cols);

    // Copy the row averages from device to host
    cudaMemcpy(h_rowAvg, d_rowAvg, rows * sizeof(double), cudaMemcpyDeviceToHost);

    CUDAMatrix resVector(rows, 1);
    resVector.SetUnderlyingMatrix(h_rowAvg);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_rowAvg);

    return resVector;
}

CUDAMatrix applyActFuncToMatrix(CUDAMatrix* inputMatrix, LayerActivationFuncs fn) {

    CUDAMatrix resMatrix = *inputMatrix;

    double* d_matrix;
    size_t size = resMatrix.GetColumns() * resMatrix.GetRows() * sizeof(double);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, size));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, resMatrix.GetUnderlyingMatrix(), size, cudaMemcpyHostToDevice));

    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((resMatrix.GetColumns() + blockSize.x - 1) / blockSize.x, (resMatrix.GetRows() + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    actFuncKernel << <gridSize, blockSize >> > (d_matrix, resMatrix.GetColumns(), resMatrix.GetRows(), fn);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy the result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(resMatrix.GetUnderlyingMatrix(), d_matrix, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_matrix));

    return resMatrix;
}

//void transposeMatrix(const float* h_in, float* h_out, int width, int height) {
CUDAMatrix transposeMatrix(CUDAMatrix inputMatrix) {
    double* d_in, * d_out;
    //size_t size = width * height * sizeof(float);
    size_t size = inputMatrix.GetRows() * inputMatrix.GetColumns() * sizeof(double);
    //double* resUnderlying = (double*)malloc(size);
    double* resUnderlying = new double[inputMatrix.GetRows() * inputMatrix.GetColumns()];

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, size));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, inputMatrix.GetUnderlyingMatrix(), size, cudaMemcpyHostToDevice));

    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((inputMatrix.GetColumns() + blockSize.x - 1) / blockSize.x, (inputMatrix.GetRows() + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    transposeKernel << <gridSize, blockSize >> > (d_out, d_in, inputMatrix.GetColumns(), inputMatrix.GetRows());
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy the result back to host
    CUDAMatrix outputMatrix(inputMatrix.GetColumns(), inputMatrix.GetRows());
    CHECK_CUDA_ERROR(cudaMemcpy(resUnderlying, d_out, size, cudaMemcpyDeviceToHost));
    outputMatrix.SetUnderlyingMatrix(resUnderlying);

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    return outputMatrix;
}

// Function to perform matrix multiplication using CUDA
//void matrixMultiply(CUDAMatrix mat1, CUDAMatrix mat2, double* matRes) {
void matrixMultiply(double* A, double* B, double* C, int mat1Rows, int mat1Cols, int mat2Rows, int mat2Cols, MatrixOperation op) {

    // Allocate memory on the GPU
    double* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, mat1Rows * mat1Cols * sizeof(double));
    cudaMalloc((void**)&d_B, mat2Rows * mat2Cols * sizeof(double));
    cudaMalloc((void**)&d_C, mat1Rows * mat2Cols * sizeof(double));

    // Copy matrices from host memory to GPU memory
    cudaMemcpy(d_A, A, mat1Rows * mat1Cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, mat2Rows * mat2Cols * sizeof(double), cudaMemcpyHostToDevice);

    // Define the block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((mat2Cols + blockSize.x - 1) / blockSize.x, (mat1Rows + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiplyKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, mat1Rows, mat1Cols, mat2Cols);

    // Copy the result matrix from GPU memory to host memory
    cudaMemcpy(C, d_C, mat1Rows * mat2Cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Function to perform matrix subtraction using CUDA
void matrixElementWiseOperations(double* A, double* B, double* C, int numRows, int numCols, MatrixOperation op) {
    // Allocate memory on the GPU
    double* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, numRows * numCols * sizeof(double));
    cudaMalloc((void**)&d_B, numRows * numCols * sizeof(double));
    cudaMalloc((void**)&d_C, numRows * numCols * sizeof(double));

    // Copy matrices from host memory to GPU memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, numRows * numCols * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, numRows * numCols * sizeof(double), cudaMemcpyHostToDevice));

    // Define the block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixElementWiseKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, numRows, numCols, op);

    // Copy the result matrix from GPU memory to host memory
    cudaMemcpy(C, d_C, numRows * numCols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrixElementWiseOperations(double* A, double constValue, double* C, int numRows, int numCols, MatrixOperation op) {
    // Allocate memory on the GPU
    double* d_A, * d_C;
    cudaMalloc((void**)&d_A, numRows * numCols * sizeof(double));
    cudaMalloc((void**)&d_C, numRows * numCols * sizeof(double));

    // Copy matrices from host memory to GPU memory
    cudaMemcpy(d_A, A, numRows * numCols * sizeof(double), cudaMemcpyHostToDevice);

    // Define the block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixElementWiseKernel << <gridSize, blockSize >> > (d_A, constValue, d_C, numRows, numCols, op);

    // Copy the result matrix from GPU memory to host memory
    cudaMemcpy(C, d_C, numRows * numCols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_C);
}

CUDAMatrix applyExpToMatrix(CUDAMatrix* inputMatrix) {
    MatrixOperation op = Exp;
    double* matRes = new double[inputMatrix->GetRows() * inputMatrix->GetColumns()];
    matrixElementWiseOperations(inputMatrix->GetUnderlyingMatrix(), -1, matRes, inputMatrix->GetRows(), inputMatrix->GetColumns(), op);
    CUDAMatrix resMatrix(inputMatrix->GetRows(), inputMatrix->GetColumns());
    resMatrix.SetUnderlyingMatrix(matRes);
    return resMatrix;
}
#pragma endregion

#pragma region public calls
CUDAMatrix CUDAMatrix::Activate(LayerActivationFuncs fn) {
    return applyActFuncToMatrix(this, fn);
}

CUDAMatrix CUDAMatrix::tanh() {
    LayerActivationFuncs actFn = tanhAct;
    return applyActFuncToMatrix(this, actFn);
}

CUDAMatrix CUDAMatrix::sigmoid() {
    LayerActivationFuncs actFn = sigAct;
    return applyActFuncToMatrix(this, actFn);
}

CUDAMatrix CUDAMatrix::exp() {
    return applyExpToMatrix(this);
}

CUDAMatrix CUDAMatrix::transpose() {
    CUDAMatrix inputMatrix = *this;
    return transposeMatrix(inputMatrix);
}

CUDAMatrix CUDAMatrix::RowAverage() {
    return RowVectorFromRowAverage(this);
}

CUDAMatrix MathHandler::TransposeMatrix(CUDAMatrix inputMatrix) {
    return transposeMatrix(inputMatrix);
}
#pragma endregion

#pragma region operator overloads
CUDAMatrix& CUDAMatrix::operator+=(const CUDAMatrix& mat2) {
    /*if (this->GetColumns() != mat2.GetColumns() || this->GetRows() != mat2.GetRows()) {
        throw std::invalid_argument("matrix addition dimensions incorrect!");
    }*/
    if (!this->_arrayForm && !this->_vectorForm && mat2._vectorForm && this->GetRows() != mat2.GetRows()) {
        throw std::invalid_argument("matrix addition by row vector dimensions incorrect!");
    }
    else if (!this->_arrayForm && !this->_vectorForm && !mat2._arrayForm && !mat2._vectorForm && this->GetColumns() != mat2.GetColumns() || this->GetRows() != mat2.GetRows()) throw std::invalid_argument("matrix addition dimensions incorrect!");

    this->_arrayForm = false;
    MatrixOperation op = Add;
    double* res = new double[this->GetRows() * this->GetColumns()];
    //matrixElementWiseOperations(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), res, this->GetRows(), mat2.GetColumns(), op);
    if (!this->_arrayForm && !this->_vectorForm && mat2._vectorForm) {
        AddMatrixAndColumnVec(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), res, this->GetRows(), this->GetColumns());
    }
    else {
        matrixElementWiseOperations(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), res, this->GetRows(), mat2.GetColumns(), op);
    }
    this->SetUnderlyingMatrix(res);
    return *this;

    //return *this + mat2;
}

CUDAMatrix CUDAMatrix::operator-=(CUDAMatrix mat2) {
    if (this->GetColumns() != mat2.GetColumns() || this->GetRows() != mat2.GetRows()) throw std::invalid_argument("matrix multiplication dimensions incorrect!");
    this->_arrayForm = false;
    MatrixOperation op = Substract;
    double* res = new double[this->GetRows() * mat2.GetColumns()];
    matrixElementWiseOperations(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), res, this->GetRows(), mat2.GetColumns(), op);
    this->SetUnderlyingMatrix(res);
    return *this;
}

CUDAMatrix CUDAMatrix::operator-(CUDAMatrix mat2) {
    if (this->GetColumns() != mat2.GetColumns() || this->GetRows() != mat2.GetRows()) throw std::invalid_argument("matrix multiplication dimensions incorrect!");

    MatrixOperation op = Substract;
    double* res = new double[this->GetRows() * mat2.GetColumns()];
    matrixElementWiseOperations(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), res, this->GetRows(), mat2.GetColumns(), op);
    CUDAMatrix resMatrix(this->GetRows(), mat2.GetColumns());
    resMatrix.SetUnderlyingMatrix(res);
    return resMatrix;
}

CUDAMatrix CUDAMatrix::operator+(CUDAMatrix mat2) {
    if (!this->_arrayForm && !this->_vectorForm && mat2._vectorForm && this->GetRows() != mat2.GetRows()) {
        throw std::invalid_argument("matrix addition by row vector dimensions incorrect!");
    }
    else if (!this->_arrayForm && !this->_vectorForm && !mat2._arrayForm && !mat2._vectorForm && this->GetColumns() != mat2.GetColumns() || this->GetRows() != mat2.GetRows()) throw std::invalid_argument("matrix addition dimensions incorrect!");

    MatrixOperation op = Add;
    double* res = new double[this->GetRows() * this->GetColumns()];
    if (!this->_arrayForm && !this->_vectorForm && mat2._vectorForm) {
        AddMatrixAndColumnVec(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), res, this->GetRows(), this->GetColumns());
    }
    else {
        matrixElementWiseOperations(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), res, this->GetRows(), mat2.GetColumns(), op);
    }
    CUDAMatrix resMatrix(this->GetRows(), this->GetColumns());
    resMatrix.SetUnderlyingMatrix(res);
    return resMatrix;
}

CUDAMatrix CUDAMatrix::operator*(const CUDAMatrix& mat2) const {
    if ((this->_arrayForm || mat2._arrayForm) && this->GetRows() != mat2.GetRows() && this->GetColumns() != mat2.GetColumns()) {
        throw std::invalid_argument("matrix multiplication in array form dimensions incorrect!");
    }

    if (!this->_arrayForm && !mat2._arrayForm && this->GetColumns() != mat2.GetRows()) {
        throw std::invalid_argument("matrix multiplication dimensions incorrect!");
    }

    MatrixOperation op = Multiply;
    double* matRes = new double[this->GetRows() * mat2.GetColumns()];

    if (this->_arrayForm || mat2._arrayForm) {
        matrixElementWiseOperations(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), matRes, this->GetRows(), mat2.GetColumns(), op);
    }
    else {
        //matrixMultiply(*this, mat2, matRes);
        matrixMultiply(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), matRes, this->GetRows(), this->GetColumns(), mat2.GetRows(), mat2.GetColumns(), op);
    }

    CUDAMatrix resMatrix(this->GetRows(), mat2.GetColumns());
    resMatrix.SetUnderlyingMatrix(matRes);
    return resMatrix;

}

CUDAMatrix CUDAMatrix::operator/(const CUDAMatrix& mat2) const {
    if (!this->_arrayForm && !mat2._arrayForm && this->GetColumns() != mat2.GetRows()) {
        throw std::invalid_argument("matrix multiplication dimensions incorrect!");
    }

    MatrixOperation op = Divide;
    double* matRes = new double[this->GetRows() * mat2.GetColumns()];

    if (this->_arrayForm && mat2._arrayForm) {
        matrixElementWiseOperations(this->GetUnderlyingMatrix(), mat2.GetUnderlyingMatrix(), matRes, this->GetRows(), mat2.GetColumns(), op);
    }
    else {
        // TODO even needed?
    }

    CUDAMatrix resMatrix(this->GetRows(), mat2.GetColumns());
    resMatrix.SetUnderlyingMatrix(matRes);
    return resMatrix;

}

CUDAMatrix CUDAMatrix::operator*(double constValue) {
    MatrixOperation op = Multiply;
    double* matRes = new double[this->GetRows() * this->GetColumns()];
    matrixElementWiseOperations(this->GetUnderlyingMatrix(), constValue, matRes, this->GetRows(), this->GetColumns(), op);
    CUDAMatrix resMatrix(this->GetRows(), this->GetColumns());
    resMatrix.SetUnderlyingMatrix(matRes);
    return resMatrix;
}

CUDAMatrix CUDAMatrix::operator+(double constValue) {
    MatrixOperation op = Add;
    double* matRes = new double[this->GetRows() * this->GetColumns()];
    matrixElementWiseOperations(this->GetUnderlyingMatrix(), constValue, matRes, this->GetRows(), this->GetColumns(), op);
    CUDAMatrix resMatrix(this->GetRows(), this->GetColumns());
    resMatrix.SetUnderlyingMatrix(matRes);
    return resMatrix;
}

CUDAMatrix CUDAMatrix::operator-(double constValue) {
    MatrixOperation op = Substract;
    double* matRes = new double[this->GetRows() * this->GetColumns()];
    matrixElementWiseOperations(this->GetUnderlyingMatrix(), constValue, matRes, this->GetRows(), this->GetColumns(), op);
    CUDAMatrix resMatrix(this->GetRows(), this->GetColumns());
    resMatrix.SetUnderlyingMatrix(matRes);
    return resMatrix;
}

CUDAMatrix CUDAMatrix::operator/(double constValue) {
    MatrixOperation op = DivideExp;
    double* matRes = new double[this->GetRows() * this->GetColumns()];
    matrixElementWiseOperations(this->GetUnderlyingMatrix(), constValue, matRes, this->GetRows(), this->GetColumns(), op);
    CUDAMatrix resMatrix(this->GetRows(), this->GetColumns());
    resMatrix.SetUnderlyingMatrix(matRes);
    return resMatrix;
}
#pragma endregion

#pragma region utility functions
double MathHandler::GenerateRandomNumber(double lowerBound, double upperBound) {
    std::uniform_real_distribution<double> unif(lowerBound, upperBound);
    return unif(_re);
}

CUDAMatrix MathHandler::ActFuncDerivative(CUDAMatrix inputMatrix, LayerActivationFuncs actFunc) {
    MatrixOperation op = None;
    if (actFunc == tanhAct) op = SquaredSubstractInvert;
    else if (actFunc == reluAct) op = ReLUDerivative;
    else if (actFunc == leakyReLU) op = LeakyReLUDerivative;
    else if (actFunc == sigAct) op = SigmoidDerivative;

    if(op == None) throw std::invalid_argument("data for act func derivative not correct!");

    double* matRes = new double[inputMatrix.GetRows() * inputMatrix.GetColumns()];
    matrixElementWiseOperations(inputMatrix.GetUnderlyingMatrix(), 1, matRes, inputMatrix.GetRows(), inputMatrix.GetColumns(), op);
    CUDAMatrix resMatrix(inputMatrix.GetRows(), inputMatrix.GetColumns());
    resMatrix.SetUnderlyingMatrix(matRes);
    return resMatrix;
}

std::vector<CUDAMatrix> MathHandler::CreateOneHotEncodedVector(std::vector<int> cmdIDs, int allClasses) {

    std::vector<CUDAMatrix> oneHotEncodedClicks;
    CUDAMatrix oneHotEncodedLabel = CUDAMatrix::Zero(allClasses, 1);

    for (int i = 0; i < cmdIDs.size(); i++)
    {
        oneHotEncodedLabel(cmdIDs[i], 0) = 1.0;
        oneHotEncodedClicks.push_back(oneHotEncodedLabel);
        oneHotEncodedLabel(cmdIDs[i], 0) = 0;
    }

    return oneHotEncodedClicks;
}

std::vector<CUDAMatrix> MathHandler::CreateBatchOneHotEncodedVector(std::vector<std::vector<int>> cmdIDs, int allClasses, int batchSize) {

    std::vector<CUDAMatrix> oneHotEncodedClicks;

    std::queue<int> randomOrder;

    std::shuffle(cmdIDs.begin(), cmdIDs.end(), _re);

    int seqLength = cmdIDs[0].size();

    for (int i = 0; i < seqLength; i++)
    {
        CUDAMatrix oneHotEncodedLabel = CUDAMatrix::Zero(allClasses, batchSize);

        for (int j = 0; j < batchSize; j++) {
            oneHotEncodedLabel(cmdIDs[j][i], j) = 1.0;
        }
        oneHotEncodedClicks.push_back(oneHotEncodedLabel);
    }

    /*for (int i = 0; i < 5; i++) {
        oneHotEncodedClicks[i].Print();
    }*/

    return oneHotEncodedClicks;
}
#pragma endregion
