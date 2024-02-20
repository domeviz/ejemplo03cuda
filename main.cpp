#include <iostream>
#include <fmt/core.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <cuda_runtime.h>

#include "glad.h"
#include <cuda_gl_interop.h>

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGTH 1080

static double x_start=-2;
static double x_end=1;
static double y_start=-1;
static double y_end=1;

static double ZOOM_FACTOR = 0.95;

#define CHECK(expr) {                       \
        auto err = (expr);                  \
        if (err != cudaSuccess) {           \
            printf("%d: %s in % s at line % d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);             \
        }                                   \
}
static int max_iterations = 10;

extern "C" void setPalette(unsigned  int* h_palette);

extern "C" void invoke_mandelbrot_kernel(
        unsigned int* buffer,
        double x_start, double x_end, double y_start, double y_end,
        int width, int height, int max_iter);

extern "C"
{
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

//--paleta
//#define PALETTE_SIZE 16
uint32_t _bswap32(uint32_t a)
{
    a = ((a & 0x000000FF) << 24) |
        ((a & 0x0000FF00) <<  8) |
        ((a & 0x00FF0000) >>  8) |
        ((a & 0xFF000000) >> 24);
    return a;
}

std::vector<unsigned int> colors_ramp = {
        _bswap32(0x000003FF),
        _bswap32(0x000003FF),
        _bswap32(0x0B0726FF),
        _bswap32(0x240B4EFF),
        _bswap32(0x410967FF),
        _bswap32(0x5D126EFF),
        _bswap32(0x781C6DFF),
        _bswap32(0x932567FF),
        _bswap32(0xAE305BFF),
        _bswap32(0xC73E4CFF),
        _bswap32(0xDC5039FF),
        _bswap32(0xED6825FF),
        _bswap32(0xF7850EFF),
        _bswap32(0xFBA40AFF),
        _bswap32(0xF9C52CFF),
        _bswap32(0xF2E660FF),
};
static bool paused= false;

GLuint textureID;
GLuint bufferID;

struct cudaGraphicsResource* cuda_pbo_resource;


void mandelbrotCuda() {
//    int threads_per_block = 1024;
//    int blocks_in_grid = std::ceil(float (IMAGE_WIDTH*IMAGE_HEIGTH)/threads_per_block);
//
//    double dx = (x_end-x_start)/IMAGE_WIDTH;
//    double dy = (y_end-y_start)/IMAGE_HEIGTH;
//
//
//    int* d_res= new int[IMAGE_HEIGTH*IMAGE_WIDTH];
//
//    invoke_mandelbrot_kernel(blocks_in_grid, threads_per_block,
//                             device_pixels_buffer, x_start, x_end,
//                             y_start, y_end, dx, dy, MAXITER );
//
//    cudaMemcpy(host_pixels_buffer, device_pixels_buffer, IMAGE_WIDTH*IMAGE_HEIGTH*4, cudaMemcpyDeviceToHost);
}

void setup_opengl(){
    glEnable(GL_TEXTURE_2D);

    //Generamos 1 buffer
    glGenBuffers(1, &bufferID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, IMAGE_WIDTH*IMAGE_HEIGTH*4, nullptr, GL_DYNAMIC_COPY);

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, bufferID, cudaGraphicsMapFlagsWriteDiscard);

    //Generamos 1 textura
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 IMAGE_WIDTH, IMAGE_HEIGTH,0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void render(){

    CHECK(cudaGraphicsMapResources(1,&cuda_pbo_resource));

    {
        uint32_t * device_pixels_buffer;
        size_t num_bytes;
        CHECK(cudaGraphicsResourceGetMappedPointer((void **)&device_pixels_buffer, &num_bytes, cuda_pbo_resource));

        invoke_mandelbrot_kernel(device_pixels_buffer, x_start, x_end, y_start, y_end,
                                 IMAGE_WIDTH, IMAGE_HEIGTH, max_iterations);

        CHECK(cudaGetLastError());

        CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo_resource));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,
                        IMAGE_WIDTH, IMAGE_HEIGTH,
                        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0,1);
            glVertex2f(-1,-1);

            glTexCoord2f(0,0);
            glVertex2f(-1,1);

            glTexCoord2f(1,0);
            glVertex2f(1,1);

            glTexCoord2f(1,1);
            glVertex2f(1,-1);
        }
        glEnd();
    }
}

void zoom(double factor) {

    double p = 0;

    if(factor==1) {

        p = 2-ZOOM_FACTOR;

    }

    else {

        p = ZOOM_FACTOR;

    }

    x_start = x_start * p;

    x_end   = x_end   * p;

    y_start = y_start * p;

    y_end   = y_end   * p;

}

void pan(double xdir, double ydir) {

    double percentx = xdir* (x_end-x_start)/100; // 10%

    double percenty = ydir * (y_end-y_start)/100; // 10%

    x_start = x_start + percentx;

    x_end   = x_end   + percentx;

    y_start = y_start + percenty;

    y_end   = y_end   + percenty;

}

int main() {

    int device =0;
    cudaSetDevice(device);
    setPalette(colors_ramp.data());

    int windowWidth = sf::VideoMode::getDesktopMode().width;
    int windowHeight = sf::VideoMode::getDesktopMode().height;

    fmt::println("Desktop resolution: {}x{}", windowWidth, windowHeight);
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Mandelbrot set");


    gladLoadGL();
    std::printf("Vendor %s, OpenGL %s, GLSL %s, Renderer %s",
                glGetString(GL_VENDOR),
                glGetString(GL_VERSION),
                glGetString(GL_SHADING_LANGUAGE_VERSION),
                glGetString(GL_RENDERER));
    setup_opengl();

    sf::Text textOptions;
    {
        textOptions.setCharacterSize(30);
        textOptions.setFillColor(sf::Color::White);
        textOptions.setStyle(sf::Text::Bold);
        textOptions.setString("OPTIONS: [R] Reset [Space] Pause");
        textOptions.setPosition(0, window.getView().getSize().y-40);
    }

    mandelbrotCuda();

    int frames =0;
    int fps = 0;
    sf::Clock clockFrames;
    // run the program as long as the window is open
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if(event.type==sf::Event::Resized) {
                textOptions.setPosition(0, window.getView().getSize().y-40);
            }
            else if(event.type==sf::Event::KeyReleased) {
                if(event.key.scancode==sf::Keyboard::Scan::Space) {
                    paused = !paused;
                }
                else if(event.key.scancode==sf::Keyboard::Scan::R) {
                    max_iterations = 10;
                    paused = false;
                }
            }
        }
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Z)) {
            zoom(1);
        }
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::X)) {
            zoom(-1);
        }
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left)) {
            pan(-1,0);
        }
        else if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right)) {
            pan(1,0);
        }
        else if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up)) {
            pan(0,1);
        }
        else if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down)) {
            pan(0,-1);
        }
        if(max_iterations<100 && !paused)
            max_iterations++;
        {
            mandelbrotCuda();

            auto msg = fmt::format("Mode=CUDA, Iterations={}, FPS: {}",
                                   max_iterations, fps);

            window.setTitle(msg);
        }

        if(clockFrames.getElapsedTime().asSeconds()>=1.0){
            fps=frames;
            frames = 0;
            clockFrames.restart();
        }

        frames++;
        window.clear(sf::Color::Black);
        render();
        window.display();

    }

    return 0;
}