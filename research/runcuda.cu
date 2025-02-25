int main() {
    int *d_matrix;
    int *h_matrix;
    int *d_transpose_matrix;

    h_matrix = (int *)malloc(N*N*sizeof(int));

    // Inicialización de la matriz de entrada
    for (int i = 0; i<N*N; i++){
        h_matrix[i]=i%N;
    }
    
    int size = N * N * sizeof(int);
   
    // Reserva de memoria en la GPU para las matrices
    CUDA_CHK(cudaMalloc((void **)&d_matrix, size));
    CUDA_CHK(cudaMalloc((void **)&d_transpose_matrix, size));

    // Copia de la matriz de entrada desde la CPU a la GPU
    CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));

    // Definición de la configuración de la grilla y los bloques
    dim3 gridSize(32, 32, 1);
    dim3 blockSize(32, 32, 1);

    for (int i=0; i<10; i++){
    // Llamada al kernel
        transposeMatrix<<<gridSize, blockSize>>>(d_matrix, d_transpose_matrix);
    
        // "Incluir inmediatamente después de la invocación al kernel."
        CUDA_CHK(cudaGetLastError());

        CUDA_CHK(cudaDeviceSynchronize());
    }   
    // Copia de la matriz transpuesta desde la GPU a la CPU
    CUDA_CHK(cudaMemcpy(h_matrix, d_transpose_matrix, size, cudaMemcpyDeviceToHost));
   
    ////////////////////////////////
    // Descomentar para imprimir  //
    ////////////////////////////////

    // printf("\n Matriz transpuesta:\n");

    // for(int i=0; i< N*N; i++){
    //     printf("%d\t", h_matrix[i]);
    // }

    // Liberación de memoria en la GPU
    CUDA_CHK(cudaFree(d_matrix));
    CUDA_CHK(cudaFree(d_transpose_matrix));

    free(h_matrix);
    return 0;
}