#include "lodepng.h"
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <future>

using std::vector, std::cout, std::endl, std::complex, std::string;
using namespace std::complex_literals;

typedef std::complex<double> cmp;

class Image
{
public:
    vector<unsigned char> pixels;
    unsigned width;
    unsigned height;

    vector<vector<unsigned char>> red;
    vector<vector<unsigned char>> green;
    vector<vector<unsigned char>> blue;
    vector<vector<unsigned char>> alpha;

    Image(unsigned width, unsigned height)
    {
        this->width = width;
        this->height = height;
    }

    Image(const char *filename, unsigned width, unsigned height)
    {
        this->width = width;
        this->height = height;

        unsigned error = lodepng::decode(pixels, width, height, filename);

        if (error)
            std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

        red.resize(height);
        green.resize(height);
        blue.resize(height);
        alpha.resize(height);

        for (size_t i = 0; i < height; i++)
        {
            red[i].reserve(width);
            green[i].reserve(width);
            blue[i].reserve(width);
            alpha[i].reserve(width);

            for (size_t j = 0; j < width * 4; j += 4)
            {
                red[i].push_back(pixels[i * 4 * width + j]);
                green[i].push_back(pixels[i * 4 * width + j + 1]);
                blue[i].push_back(pixels[i * 4 * width + j + 2]);
                alpha[i].push_back(pixels[i * 4 * width + j + 3]);
            }
        }
    }

    void Save(const char *filename)
    {
        unsigned error = lodepng::encode(filename, pixels, width, height);
        if (error)
            std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }

    // this pads a specific RGBA channel with zeros so the length of the vector will be a power of 2 (to prepare it for the FFT algorithm)
    void PadChannelWithZeros(vector<vector<unsigned char>> &channel, unsigned newWidth, unsigned newHeight)
    {
        for (vector<unsigned char> &row : channel)
        {
            row.resize(newWidth);
        }

        for (size_t i = 0; i < newHeight - height; i++)
        {
            channel.push_back(vector<unsigned char>(newWidth, 0));
        }
    }

    void PadImageWithZeros()
    {
        unsigned newWidth = 1;
        while (newWidth < width)
        {
            newWidth <<= 1;
        }

        unsigned newHeight = 1;
        while (newHeight < height)
        {
            newHeight <<= 1;
        }

        PadChannelWithZeros(red, newWidth, newHeight);
        PadChannelWithZeros(green, newWidth, newHeight);
        PadChannelWithZeros(blue, newWidth, newHeight);
        PadChannelWithZeros(alpha, newWidth, newHeight);

        width = newWidth;
        height = newHeight;
    }

    // removes extra padding zeroes from RGBA channel after Inverse FFT has been applied
    void UnpadChannel(vector<vector<unsigned char>> &channel, unsigned originalWidth, unsigned originalHeight)
    {
        channel.resize(originalHeight);
        for (vector<unsigned char> &row : channel)
        {
            row.resize(originalWidth);
        }
    }

    void UnpadImage(unsigned originalWidth, unsigned originalHeight)
    {
        UnpadChannel(red, originalWidth, originalHeight);
        UnpadChannel(green, originalWidth, originalHeight);
        UnpadChannel(blue, originalWidth, originalHeight);
        UnpadChannel(alpha, originalWidth, originalHeight);
    }
};

const double PI = 3.14159265359879;

class ImageFourierTransform
{
public:
    unsigned width;
    unsigned height;

    unsigned originalImageWidth;
    unsigned originalImageHeight;

    vector<vector<cmp>> red;
    vector<vector<cmp>> green;
    vector<vector<cmp>> blue;
    vector<vector<cmp>> alpha;

    std::pair<vector<cmp>, vector<cmp>> even_odd(vector<cmp> values)
    {
        size_t N = values.size();
        vector<cmp> even;
        vector<cmp> odd;
        even.reserve(N / 2);
        odd.reserve(N / 2);

        for (size_t i = 0; i < N; i++)
        {
            if (i % 2 == 0)
            {
                even.push_back(values[i]);
            }
            else
            {
                odd.push_back(values[i]);
            }
        }

        return std::pair(even, odd);
    }

    template <class T>
    vector<cmp> ToCmpVector(const vector<T> &values)
    {
        vector<cmp> vec;
        vec.reserve(values.size());
        for (const T &c : values)
        {
            vec.push_back(cmp(c, 0));
        }

        return vec;
    }

    vector<cmp> FFT1d(const vector<cmp> &values, size_t n, bool isInverse = false)
    {
        if (n == 1)
        {
            return values;
        }
        else
        {
            auto [firstHalf, secondHalf] = even_odd(values);
            firstHalf = FFT1d(firstHalf, n / 2, isInverse);
            secondHalf = FFT1d(secondHalf, n / 2, isInverse);

            vector<cmp> ret;
            ret.resize(n);
            for (size_t k = 0; k < n / 2; k++)
            {
                cmp p = firstHalf[k];
                double angle = 2 * PI * k / n;
                cmp q = secondHalf[k] * exp(cmp(0, (isInverse ? 1 : -1)) * angle);
                ret[k] = p + q;
                ret[k + (n / 2)] = p - q;
            }

            return ret;
        }
    }

    vector<cmp> FFT1d(const vector<unsigned char> &values, bool isInverse = false)
    {
        vector<cmp> cmpValues = ToCmpVector(values);
        return FFT1d(cmpValues, cmpValues.size(), isInverse);
    }

    vector<cmp> FFT1d(const vector<int> &values, bool isInverse = false)
    {
        vector<cmp> cmpValues = ToCmpVector(values);
        return FFT1d(cmpValues, cmpValues.size(), isInverse);
    }

    vector<cmp> FFT1d(const vector<cmp> &cmpValues, bool isInverse = false)
    {
        return FFT1d(cmpValues, cmpValues.size(), isInverse);
    }

    vector<cmp> FFT1dInverse(const vector<cmp> &values)
    {
        vector<cmp> cmpResult = FFT1d(values, true);
        size_t N = cmpResult.size();

        for (cmp &c : cmpResult)
        {
            c /= N;
        }

        return cmpResult;
    }

    static void Transpose(vector<vector<cmp>> &matrix)
    {
        if (matrix.size() == matrix[0].size())
        {
            for (size_t row = 0; row < matrix.size(); row++)
            {
                for (size_t col = 0; col < row; col++)
                {
                    std::swap(matrix[row][col], matrix[col][row]);
                }
            }
        }
        else
        {
            vector<vector<cmp>> tMatrix;
            size_t numRows = matrix.size();
            size_t numCols = matrix[0].size();

            tMatrix.resize(numCols);
            for (size_t col = 0; col < numCols; col++)
            {
                tMatrix[col].resize(numRows);
                for (size_t row = 0; row < numRows; row++)
                {
                    tMatrix[col][row] = matrix[row][col];
                }
            }

            matrix = tMatrix;
        }
    }
    void FFT2d(vector<vector<cmp>> &out, const vector<vector<unsigned char>> &values)
    {
        // multiply input values by -1^(row + col) so that the spectral representation will be centered and filtering can be done easily
        vector<vector<int>> valuesPhaseShift;
        valuesPhaseShift.resize(values.size());
        for (size_t i = 0; i < values.size(); i++)
        {
            for (size_t j = 0; j < values[0].size(); j++)
            {

                valuesPhaseShift[i].push_back(((i + j) % 2 == 0) ? values[i][j] : -values[i][j]);
            }
        }
        out.resize(height);
        for (size_t row_index = 0; row_index < height; row_index++)
        {
            out[row_index] = FFT1d(valuesPhaseShift.at(row_index));
        }

        Transpose(out);

        for (size_t col_index = 0; col_index < width; col_index++)
        {
            out[col_index] = FFT1d(out.at(col_index));
        }

        Transpose(out);
    }

    vector<vector<unsigned char>> InverseFFT2d(vector<vector<cmp>> &values)
    {
        vector<vector<cmp>> cmpOut;
        cmpOut.resize(width);

        Transpose(values);
        for (size_t col_index = 0; col_index < width; col_index++)
        {
            cmpOut[col_index] = FFT1dInverse(values.at(col_index));
        }

        Transpose(values);
        Transpose(cmpOut);
        for (size_t row_index = 0; row_index < height; row_index++)
        {
            cmpOut[row_index] = FFT1dInverse(cmpOut.at(row_index));
        }

        vector<vector<unsigned char>> out;

        out.resize(height);
        for (size_t row = 0; row < height; row++)
        {
            out[row].reserve(width);
            for (size_t col = 0; col < width; col++)
            {
                out[row].push_back(static_cast<unsigned char>(abs(cmpOut[row][col].real())));
            }
        }
        return out;
    }

    ImageFourierTransform() {}

    ImageFourierTransform(Image &image)
    {
        originalImageWidth = image.width;
        originalImageHeight = image.height;

        image.PadImageWithZeros();

        width = image.width;
        height = image.height;

        FFT2d(red, image.red);
        FFT2d(green, image.green);
        FFT2d(blue, image.blue);
        FFT2d(alpha, image.alpha);
    }

    Image ToImage()
    {
        Image image(width, height);

        image.red = InverseFFT2d(red);
        image.green = InverseFFT2d(green);
        image.blue = InverseFFT2d(blue);
        image.alpha = InverseFFT2d(alpha);

        image.UnpadImage(originalImageWidth, originalImageHeight);
        image.width = originalImageWidth;
        image.height = originalImageHeight;

        image.pixels = vector<unsigned char>();
        image.pixels.reserve(originalImageWidth * originalImageHeight * 4);

        for (size_t row = 0; row < originalImageHeight; row++)
        {
            for (size_t col = 0; col < originalImageWidth; col++)
            {
                image.pixels.push_back(image.red[row][col]);
                image.pixels.push_back(image.green[row][col]);
                image.pixels.push_back(image.blue[row][col]);
                image.pixels.push_back(image.alpha[row][col]);
            }
        }
        return image;
    }

    void FilterChannel(vector<vector<cmp>> &values, double (*filterFunc)(int, int))
    {
        for (size_t row = 0; row < height; row++)
        {
            for (size_t col = 0; col < width; col++)
            {
                values[row][col] *= filterFunc(row - height / 2, col - width / 2);
            }
        }
    }

    void Filter(double (*filterFunc)(int, int))
    {
        FilterChannel(red, filterFunc);
        FilterChannel(green, filterFunc);
        FilterChannel(blue, filterFunc);
    }
};

double BPF(double hpAmount, double lpAmount, int centeredRow, int centeredCol)
{
    double dist = sqrt(centeredRow * centeredRow + centeredCol * centeredCol);

    if (dist < hpAmount || dist > lpAmount)
    {
        return 0;
    }
    return 1;
}

void TestFFT()
{
    vector<vector<unsigned char>> test = {{245, 2, 1, 0}, {30, 172, 5, 19}, {1, 2, 3, 4}, {180, 190, 200, 210}};
    ImageFourierTransform testFFT;
    testFFT.width = 4;
    testFFT.height = 4;

    vector<vector<cmp>> testFFTCoefs;
    testFFT.FFT2d(testFFTCoefs, test);

    vector<vector<unsigned char>> res = testFFT.InverseFFT2d(testFFTCoefs);
    for (size_t i = 0; i < testFFT.height; i++)
    {
        for (size_t j = 0; j < testFFT.width; j++)
        {
            if (test[i][j] != res[i][j])
            {
                cout << "Error: FFT algorithm not working properly";
                return;
            }
        }
    }
}

void TestImageParsing()
{
    const char *fileName = "animated_cat.png";
    Image cat(fileName, 256, 256);

    vector<int> pixel_vec;
    vector<int> red_vec;
    for (size_t i = 102800; i < 102840; i += 4)
    {
        pixel_vec.push_back((int)cat.pixels[i]);
    }

    for (size_t i = 100; i < 110; i++)
    {
        red_vec.push_back((int)cat.red[100][i]);
    }

    for (size_t i = 0; i < pixel_vec.size(); i++)
    {
        if (pixel_vec[i] != red_vec[i])
        {
            cout << "Error: Image parsing into RGBA channels not working properly";
            return;
        }
    }
}

int main()
{

    TestFFT();
    TestImageParsing();

    const char *fileName = "__example__png__photo.png";
    Image image(fileName, 1024, 1024); // put in width and height in pixels of the image
    cout << "Parsed input Image" << endl;
    ImageFourierTransform imageFFT(image);
    cout << "FFT Completed" << endl;

    imageFFT.Filter([](int row, int col)
                    { return BPF(150, 400, row, col); });
    cout << "Filter Completed" << endl;

    Image imageFiltered = imageFFT.ToImage();

    cout << "Converted Back To Image" << endl;
    imageFiltered.Save("__filtered__example__image.png");
    return 0;
}