#include <CL/sycl.hpp>
#include <SDL2/SDL.h>
#include <iostream>
#include <array>
using namespace std;

// SDL elements
SDL_Window* win = NULL;
SDL_Surface* surface = NULL;
SDL_Renderer* render = NULL;

// Config
const int HEIGHT = 800;
const int WIDTH = 1200;
const int MAXITER = 500;

// Denotes view area of mandelbrot
float xStart = -2.0;
float xFinish = 1.0;
float yStart = -1.0;
float yFinish = 1.0;

// For zooming
bool isClicking = false;
array<int, 2> initialMousePos = {0, 0};
array<int, 2> finalMousePos = {0, 0};

// Array of hues for each pixel
array<int, HEIGHT * WIDTH> hue = {};
// Buffer for hue array
// Passed into kernel as 1D array (contiguous) but interpreted as 2D in draw() func
auto hueBuf = sycl::buffer<int>(hue.data(), WIDTH * HEIGHT);
// Saturation is constant
// Array of vales for each pixel
array<int, HEIGHT * WIDTH> value = {};
// Buffer for value array
auto valueBuf = sycl::buffer<int>(value.data(), WIDTH * HEIGHT);

// SYCL GPU queue 
sycl::queue mandelbrotQueue{sycl::gpu_selector()};

// Helper function converts hsv to rgb (SDL can only render RGB colours)
array<float, 3> HSVtoRGB(float h, float s, float v) {
    float c = v * s;
    float scaledH = fmod(h / 60.0, 6);
    float x = c * (1 - fabs(fmod(scaledH, 2) - 1));
    float m = v - c;

    array<float, 3> rgb = {};
    
    if (0 <= scaledH && scaledH < 1) { rgb[0] = c; rgb[1] = x; rgb[2] = 0; }
    else if (1 <= scaledH && scaledH < 2) { rgb[0] = x; rgb[1] = c; rgb[2] = 0; }
    else if (2 <= scaledH && scaledH < 3) { rgb[0] = 0; rgb[1] = c; rgb[2] = x; }
    else if (3 <= scaledH && scaledH < 4) { rgb[0] = 0; rgb[1] = x; rgb[2] = c; }
    else if (4 <= scaledH && scaledH < 5) { rgb[0] = x; rgb[1] = 0; rgb[2] = c; }
    else if (5 <= scaledH && scaledH < 6) { rgb[0] = c; rgb[1] = 0; rgb[2] = x; }
    else { rgb[0] = 0; rgb[1] = 0; rgb[2] = 0; }

    return rgb;
}

// Calculates divergence of function f_c(z) = z^2 + c
int mandelbrot(complex<float> c) {
    complex<float> funcZ = 0;
    int count = 0;
    // Limited by MAXITER to prevent infinite looping
    while (abs(funcZ) <= 2 && count < MAXITER) {
        funcZ = funcZ * funcZ + c;
        count += 1;
    }
    return count;
}

void collectMandelbrotVals(float x1, float x2, float y1, float y2) {
    // Enqueue task
    mandelbrotQueue.submit([&](sycl::handler& cgh) {
        // Accessors
        auto hueAcc = hueBuf.get_access<sycl::access::mode::write>(cgh);
        auto valueAcc = valueBuf.get_access<sycl::access::mode::write>(cgh);

        // Pixel hues and values are calculated in parallel
        cgh.parallel_for(sycl::range<2>{WIDTH, HEIGHT}, [=](sycl::item<2> index) {
            // Convert polar coordinates into complex number
            int xCoord = index[0]; int yCoord = index[1];
            float x = x1 + (xCoord * (x2 - x1) / (WIDTH - 1));
            float y = y2 - (yCoord * (y2 - y1) / (HEIGHT - 1));
            complex<float> polarCoord(x, y);

            // Calculate divergence
            int m = mandelbrot(polarCoord);
            // Adjust hue and value accordingly
            int hue = int(255 * m / MAXITER);
            // Write results to accessors
            valueAcc[xCoord + (yCoord * WIDTH)] = m < MAXITER ? 1 : 0;
            hueAcc[xCoord + (yCoord * WIDTH)] = hue;
        });
    });

    // Buffers write back to global memory
    hueBuf.get_access<sycl::access::mode::read>();
    valueBuf.get_access<sycl::access::mode::read>();
}

void init() {
    // Initialize SDL
    SDL_Init(SDL_INIT_VIDEO);
    // Create window
    win = SDL_CreateWindow("Mandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
}

void draw() {
    SDL_SetRenderDrawColor(render, 255, 255, 255, 255);
    SDL_RenderClear(render);

    // Draw each pixel, using helper function to convert from hsv to rgb
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        array<float, 3> rgb = HSVtoRGB(hue[i], 1, value[i]);
        SDL_SetRenderDrawColor(render, rgb[0] * 255, rgb[1] * 255, rgb[2] * 255, 255);
        SDL_RenderDrawPoint(render, i % WIDTH, i / WIDTH);
    }

    // Draw viewfinder rect if user is zooming
    if (isClicking) {
        SDL_Rect r;
        r.x = (initialMousePos[0] > finalMousePos[0]) ? finalMousePos[0] : initialMousePos[0];
        r.y = (initialMousePos[1] > finalMousePos[1]) ? finalMousePos[1] : initialMousePos[1];
        r.w = abs(initialMousePos[0] - finalMousePos[0]);
        float multiplier = float(abs(initialMousePos[0] - finalMousePos[0])) / WIDTH;
        r.h = HEIGHT * multiplier;

        SDL_SetRenderDrawColor(render, 0, 0, 0, 255);
        SDL_RenderDrawRect(render, &r);

        r.x += 1; r.y += 1; r.w -= 2; r.h -= 2;
        SDL_SetRenderDrawColor(render, 255, 255, 255, 255);
        SDL_RenderDrawRect(render, &r); 
    }

    SDL_RenderPresent(render);
}

// Free resources and close SDL
void close() {
    SDL_DestroyWindow(win);
    SDL_Quit();
}

int main() {
    init();
    render = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    collectMandelbrotVals(xStart, xFinish, yStart, yFinish);
    draw();

    bool isQuit = false;
    isClicking = false;
    SDL_Event event;
    while (!isQuit) {
        if (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                isQuit = true;
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN) {
                isClicking = true;
                initialMousePos[0] = event.motion.x;
                initialMousePos[1] = event.motion.y;
            }
            else if (event.type == SDL_MOUSEBUTTONUP) {
                finalMousePos[0] = event.motion.x;
                finalMousePos[1] = event.motion.y;
                draw();
                
                // Calculate new viewfinder range based on zoom
                int startX = (initialMousePos[0] > finalMousePos[0]) ? finalMousePos[0] : initialMousePos[0];
                int startY = (initialMousePos[1] > finalMousePos[1]) ? finalMousePos[1] : initialMousePos[1];

                int w = abs(finalMousePos[0] - initialMousePos[0]);
                
                float scaleW = float(xFinish - xStart) / float(WIDTH);
                float scaleH = float(yFinish - yStart) / float(HEIGHT);

                float newWidth = float(w) * float(scaleW);
                float newHeight = float((HEIGHT * (float(w) / WIDTH)) * float(scaleH));

                xStart += float(startX) * float(scaleW);
                xFinish = float(xStart) + newWidth;

                yFinish -= float(startY) * float(scaleH);
                yStart = float(yFinish) - newHeight;
                
                // Re-render mandelbrot
                collectMandelbrotVals(xStart, xFinish, yStart, yFinish);
                isClicking = false;
                draw();
            }
        }
    }

    close();

    return 0;
}