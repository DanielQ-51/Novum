#pragma once

#include "util.cuh"
#include <vector>

class Image 
{
private:
    float4 toneMap(float4 c);
    float4 aces_fitted(float4 c);
    float4 gammaCorrect(float4 c);
    std::vector<float4> postProcessImage();
public:
    bool postProcess;
    bool use_fitted_aces = true;
    Image(int w, int h);
    ~Image();

    void setColor(int x, int y, float4 c);
    float4 getColor(int x, int y);
    void saveImageBMP(std::string fileName);
    void saveImageCSV();
    void saveImageCSV_MONO(int choice);
    

    int toIndex(int x, int y);
    std::pair<int,int> fromIndex(int i);

    std::vector<float4> data();

    const int width, height;
    std::vector<float4> pixels;
};

Image loadBMPToImage(const std::string &filename, bool isData);