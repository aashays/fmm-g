void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}



int devcheck(int gpudevice)
{
    int device_count=0;
    int device;  // used with cudaGetDevice() to verify cudaSetDevice()

    // get the number of non-emulation devices detected
    cudaGetDeviceCount( &device_count);
    if (gpudevice >= device_count)
    {
        printf("gpudevice = %d , valid devices = [ ", gpudevice);
   for (int i=0; i<device_count; i++)
      printf("%d ", i);
   printf("] ... exiting \n");
        exit(1);
    }
    cudaError_t cudareturn;
    cudaDeviceProp deviceProp;

    // cudaGetDeviceProperties() is also demonstrated in the deviceQuery/ 
    // of the sdk projects directory
    cudaGetDeviceProperties(&deviceProp, gpudevice);
    printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n",
     deviceProp.major, deviceProp.minor);

    if (deviceProp.major > 999)
    {
        printf("warning, CUDA Device Emulation (CPU) detected, exiting\n");
                exit(1);
    }

    // choose a cuda device for kernel execution
    cudareturn=cudaSetDevice(gpudevice);
    if (cudareturn == cudaErrorInvalidDevice)
   {
        perror("cudaSetDevice returned cudaErrorInvalidDevice");
    }
    else
    {
        // double check that device was properly selected
        cudaGetDevice(&device);
        printf("cudaGetDevice()=%d\n",device);
    }
    return(0);
}

