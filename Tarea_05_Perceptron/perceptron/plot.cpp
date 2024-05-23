#include "CImg.h"
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <iostream>

using namespace cimg_library;
using std::vector;

std::string
exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);

    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

int main() {

    int epochs = 100;
    vector<float> x1 = {10, 100, 1000};   
    // vector<float> x1 = {10000, 20000, 30000, 40000, 50000, 60000};   
    vector<float> y;

    for (int i: x1) {
            std::string command1 = "./train" + std::string(" ") + std::to_string(epochs) + " " + std::to_string(i);
            std::system(command1.c_str());
            std::string command2 = "./test";
            std::string output = exec(command2.c_str());
            y.push_back(std::stod(output));

            printf("%d, %f\n", i, std::stod(output));
    }

    std::vector<float> x = {0, 1, 2};

    if (x.size() != y.size()) {
        std::cerr << "Error: x and y arrays must have the same size." << std::endl;
        return -1;
    }

    const int width = 600;
    const int height = 400;
    const int margin = 50;

    CImg<unsigned char> img(width, height, 1, 3, 255);

    img.fill(255);

    const unsigned char black[] = {0, 0, 0};
    const unsigned char red[] = {255, 0, 0};

    img.draw_line(margin, height - margin, width - margin, height - margin, black); // X-axis
    img.draw_line(margin, margin, margin, height - margin, black); // Y-axis
                                                                   //    // Draw x-axis labels
    float x_max = *std::max_element(x1.begin(), x1.end());
    float y_max = *std::max_element(y.begin(), y.end());

    for (int i = 0; i <= 5; ++i) {
        int x_pos = margin + static_cast<int>((i * x_max / 5) / x_max * (width - 2 * margin));
        std::string label = std::to_string(static_cast<int>(i * x_max / 5));
        img.draw_text(x_pos - 10, height - margin + 5, label.c_str(), black);
    }

    // Draw y-axis labels
    for (int i = 0; i <= 5; ++i) {
        int y_pos = height - margin - static_cast<int>((i * y_max / 5) / y_max * (height - 2 * margin));
        std::string label = std::to_string(static_cast<int>(i * y_max / 5));
        img.draw_text(margin - 40, y_pos - 5, label.c_str(), black);
    }

    for (size_t i = 0; i < x.size(); ++i) {
        int x_pos = margin + static_cast<int>((x[i] / x.back()) * (width - 2 * margin));
        int y_pos = height - margin - static_cast<int>((y[i] / *y.begin()) * (height - 2 * margin));

        img.draw_circle(x_pos, y_pos, 3, red);

        if (i > 0) {
            int x_prev = margin + static_cast<int>((x[i - 1] / x.back()) * (width - 2 * margin));
            int y_prev = height - margin - static_cast<int>((y[i - 1] / *y.begin()) * (height - 2 * margin));
            img.draw_line(x_prev, y_prev, x_pos, y_pos, red);
        }
    }

    CImgDisplay main_display(img, "2D Chart");

    while (!main_display.is_closed()) {
        main_display.wait();
    }
    return 0;
}

