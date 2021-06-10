#ifdef FPGA

#include <stdio.h>
#include <errno.h>

#include <CL/opencl.h>

#include "gemm_fpga.h"

#define KERNEL_FILE "gemm.aocx"
#define KERNEL_NAME "gemm_nn"

extern void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);

static void checkError(cl_int err, const char *name);

static int GEMM_FPGA_INITED;

// OpenCL runtime configuration
static cl_device_id device;
static cl_context context;
static cl_program program;
static cl_command_queue queue;

static cl_kernel kernel;

void gemm_fpga_init()
{
    cl_int status;
    cl_platform_id platform;

    if (GEMM_FPGA_INITED)
        return;

    // Identify a platform
    status = clGetPlatformIDs(1, &platform, NULL);
    checkError(status, "Failed to get Platform ID");

    // Access a device
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    checkError(status, "Failed to get Device ID");

    // Create the context.
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkError(status, "Failed to create the context");

    // Create the program for the device.
    const unsigned char *source_str;
    size_t source_size;

    // Load the source code containing the kernel
    FILE *fp = fopen (KERNEL_FILE, "r");
    if (!fp) {
        printf("Error: %s fopen failed, err %d", KERNEL_FILE, errno);
        checkError(CL_INVALID_VALUE, "");
    }

    source_str = (const unsigned char *) malloc (0x5000000);
    source_size = fread ((void*)source_str, 1, 0x5000000, fp);
    fclose (fp);

    // Create Program.
    cl_int _status;
    program = clCreateProgramWithBinary(context, 1, &device, &source_size,
            (const unsigned char **) &source_str, &_status, &status);
    checkError(status, "Failed to create program");

    // Build the program that was just created.
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    checkError(status, "Failed to build program");

    // Create Command Queue.
    queue = clCreateCommandQueue(context, device, 0, &status);
    checkError(status, "Failed to create command queue");

    // Create Kernel.
    const char *kernel_name = KERNEL_NAME;
    kernel = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    GEMM_FPGA_INITED = 1;
    return;
}

void gemm_fpga_deinit()
{
    if (!GEMM_FPGA_INITED)
        return;

    if(kernel) {
        clReleaseKernel(kernel);
    }

    if(queue) {
        clReleaseCommandQueue(queue);
    }

    if(program) {
        clReleaseProgram(program);
    }

    if(context) {
        clReleaseContext(context);
    }

    GEMM_FPGA_INITED = 0;
    return;
}

void gemm_nn_fpga(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    if (!GEMM_FPGA_INITED) {
        printf("Warning: Gemm FPGA wasn't initialized (Using CPU).\n");
        return gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }

    cl_int status;

    cl_event kernel_event;
    cl_event finish_event;

    // Create buffer objects
    cl_mem a_buf = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
        M * K * sizeof(float), A, &status);
    checkError(status, "Failed to create buffer for input A");

    cl_mem b_buf = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
        K * N * sizeof(float), B, &status);
    checkError(status, "Failed to create buffer for input B");

    cl_mem c_buf = clCreateBuffer(context,  CL_MEM_COPY_HOST_PTR, 
        M * N * sizeof(float), C, &status);
    checkError(status, "Failed to create buffer for input/output C");

    // Pass arguments to the kernel
    unsigned argi = 0;
    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *) &M);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *) &N);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *) &K);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(float), (void *) &ALPHA);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), (void *) &a_buf);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *) &lda);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), (void *) &b_buf);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *) &ldb);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), (void *) &c_buf);
    checkError(status, "Failed to set argument");

    status = clSetKernelArg(kernel, argi++, sizeof(int), (void *) &ldc);
    checkError(status, "Failed to set argument");

    // Enqueues a command to execute a kernel on a device
    const size_t global_work_size = 1;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        &global_work_size, NULL, 0, NULL, &kernel_event);
    checkError(status, "Failed to launch kernel");

    // Enqueue commands to read from a buffer object to host memory
    status = clEnqueueReadBuffer(queue, c_buf, CL_FALSE,
        0, M * N * sizeof(float), C, 1, &kernel_event, &finish_event);
    checkError(status, "Failed to read from a buffer object");

    clWaitForEvents(1, &finish_event);

    clReleaseEvent(kernel_event);
    clReleaseEvent(finish_event);

    // Release buffer objects
    if(a_buf) {
        clReleaseMemObject(a_buf);
    }
    if(b_buf) {
        clReleaseMemObject(b_buf);
    }
    if(c_buf) {
        clReleaseMemObject(c_buf);
    }

    return;
}

// ====== Helper functions ======
static void checkError(cl_int err, const char *name)
{
    if(err != CL_SUCCESS)
    {
        printf("Error: %s %d",name,err);
        switch(err)
        {
            case CL_DEVICE_NOT_FOUND :printf("(CL_DEVICE_NOT_FOUND)");break;
            case CL_DEVICE_NOT_AVAILABLE :printf("(CL_DEVICE_NOT_AVAILABLE)");break;
            case CL_COMPILER_NOT_AVAILABLE :printf("(CL_COMPILER_NOT_AVAILABLE)");break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE :printf("(CL_MEM_OBJECT_ALIOCATION_FAILURE)");break;
            case CL_OUT_OF_RESOURCES :printf("(CL_OUT_OF_RESOURCES)");break;
            case CL_OUT_OF_HOST_MEMORY :printf("(CL_OUT_OF_HOST_MEMORY)");break;
            case CL_MEM_COPY_OVERLAP :printf("(CL_MEM_COPY_OVERLAP)");break;
            case CL_BUILD_PROGRAM_FAILURE:printf("(CL_BUILD_PROGRAM_FAILURE)");break;
            case CL_INVALID_VALUE:printf("(CL_INVALID_VALUE)");break;
            case CL_INVALID_DEVICE_TYPE:printf("(CL_INVALID_DEVICE_TYPE)");break;
            case CL_INVALID_DEVICE:printf("(CL_INVALID_DEVICE)");break;
            case CL_INVALID_CONTEXT:printf("(CL_INVALID_CONTEXT)");break;
            case CL_INVALID_BINARY:printf("(CL_INVALID_BINARY)");break;
            case CL_INVALID_BUILD_OPTIONS:printf("(CL_INVALID_BUILD_OPTIONS)");break;
            case CL_INVALID_PROGRAM:printf("(CL_INVALID_PROGRAM)");break;
            case CL_INVALID_PROGRAM_EXECUTABLE:printf("(CL_INVALID_PROGRAM_EXECUTABLE)");break;
            case CL_INVALID_KERNEL_DEFINITION:printf("(CL_INVALID_KERNEL_DEFINITION)");break;
            case CL_INVALID_KERNEL:printf("(CL_INVALID_KERNEL)");break;
            case CL_INVALID_KERNEL_ARGS:printf("(CL_INVALID_KERNEL_ARGS)");break;
            case CL_INVALID_OPERATION:printf("(CL_INVALID_OPERATION)");break;
            case CL_INVALID_COMMAND_QUEUE:printf("(CL_INVALID_COMMAND_QUEUE)");break;
            case CL_INVALID_WORK_DIMENSION:printf("(CL_INVALID_WORK_DIMENSION)");break;
            case CL_INVALID_WORK_GROUP_SIZE:printf("(CL_INVALID_WORK_GROUP_SIZE)");break;
            case CL_INVALID_WORK_ITEM_SIZE:printf("(CL_INVALID_WORK_ITEM_SIZE)");break;
            case CL_INVALID_GLOBAL_WORK_SIZE:printf("(CL_INVALID_GLOBAL_WORK_SIZE)");break;
            case CL_INVALID_GLOBAL_OFFSET:printf("(CL_INVALID_GLOBAL_OFFSET)");break;
            case CL_INVALID_IMAGE_SIZE:printf("(CL_INVALID_IMAGE_SIZE)");break;
            case CL_INVALID_EVENT_WAIT_LIST:printf("(CL_INVALID_EVENT_WAIT_LIST)");break;
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:printf("(CL_MISALIGNED_SUB_BUFFER_OFFSET)");break;

            default:
                                                 break;
        }
        printf("\n");
        exit(1);
    }
}

#endif
