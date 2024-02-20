__constant__ unsigned int c_palette[16];

__device__ int diverge(double cx, double cy, int max_iters) {
    int iter = 0;
    double vx=cx;
    double vy = cy;
    double tx, ty;

    while(iter<max_iters && (vx*vx+vy*vy)<4) {
        tx = vx*vx-vy*vy + cx;
        ty = 2 *vx*vy+cy;

        vx = tx;
        vy = ty;

        iter++;
    }

    if(iter>0 && iter<max_iters) {
        //return  0xFFFFFF;
        return c_palette[iter%16];
    }

    return 0x000000;
}

__global__ void mandelbrot_kernel(unsigned int* buffer,
                                  double x_start, double x_end,double y_start, double y_end,
                                  double dx, double dy,
                                  int width, int height,
                                  int max_iter) {
    int id = blockDim.x*blockIdx.x + threadIdx.x;

    int i = id % width;
    int j = id / width;

    //cx+cxi numero complejo
    double cx = x_start+i*dx;
    double cy = y_end-j*dy;

    int color = diverge(cx, cy, max_iter);

    buffer[id] =  color;
}

//--exportar
extern "C" void setPalette(unsigned int* h_palette) {
    cudaMemcpyToSymbol(c_palette, h_palette, 16*sizeof(unsigned int));
}

extern "C" void invoke_mandelbrot_kernel(
        unsigned int* buffer,
        double x_start, double x_end, double y_start, double y_end,
        int width, int height,
        int max_iter) {

    int threads_per_block = 1024;
    int blocks_in_grid = std::ceil(float(width*height) / threads_per_block);

    double dx = (x_end - x_start)/(width - 1);
    double dy = (y_end - y_start)/(height - 1);

    mandelbrot_kernel<<<blocks_in_grid, threads_per_block>>>(
            buffer, x_start, x_end, y_start, y_end,
            dx, dy,
            width, height,
            max_iter
    );
}