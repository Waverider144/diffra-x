#include "diffrax/optimizer.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace diffrax {
namespace {

constexpr float kPi = 3.14159265358979323846F;

void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << file << ':' << line << " CUDA call " << expression << " failed: "
                << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

void cufft_check(cufftResult status, const char* expression, const char* file, int line) {
    if (status != CUFFT_SUCCESS) {
        std::ostringstream message;
        message << file << ':' << line << " cuFFT call " << expression
                << " failed with code " << static_cast<int>(status);
        throw std::runtime_error(message.str());
    }
}

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)
#define CUFFT_CHECK(call) cufft_check((call), #call, __FILE__, __LINE__)

template <typename T>
T read_or(const cv::FileNode& node, const char* key, T fallback) {
    const cv::FileNode value = node[key];
    return value.empty() ? fallback : static_cast<T>(value);
}

std::string read_string_or(const cv::FileNode& node, const char* key, const std::string& fallback) {
    const cv::FileNode value = node[key];
    return value.empty() ? fallback : static_cast<std::string>(value);
}

struct SourcePoint {
    float fx;
    float fy;
    float weight;
};

__device__ cufftComplex complex_mul(cufftComplex a, cufftComplex b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__global__ void phase_to_padded_kernel(const float* phase, cufftComplex* padded,
                                        int h, int w, int ph, int pw, int oy, int ox) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ph * pw;
    if (index >= total) return;
    const int y = index / pw;
    const int x = index % pw;
    cufftComplex value{0.0F, 0.0F};
    if (y >= oy && y < oy + h && x >= ox && x < ox + w) {
        const float p = phase[(y - oy) * w + (x - ox)];
        sincosf(p, &value.y, &value.x);
    }
    padded[index] = value;
}

__global__ void transfer_kernel(cufftComplex* transfer, int h, int w, float dx,
                                float wavelength, float distance) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= h * w) return;
    const int y = index / w;
    const int x = index % w;
    const float fx = (x <= w / 2 ? x : x - w) / (w * dx);
    const float fy = (y <= h / 2 ? y : y - h) / (h * dx);
    const float term = 1.0F / (wavelength * wavelength) - fx * fx - fy * fy;
    const float fx_max = dx * w / (2.0F * wavelength * sqrtf((dx * w) * (dx * w) + distance * distance));
    const float fy_max = dx * h / (2.0F * wavelength * sqrtf((dx * h) * (dx * h) + distance * distance));
    if (term <= 0.0F || fabsf(fx) >= fx_max || fabsf(fy) >= fy_max) {
        transfer[index] = {0.0F, 0.0F};
        return;
    }
    const float phase = 2.0F * kPi * sqrtf(term) * distance;
    sincosf(phase, &transfer[index].y, &transfer[index].x);
}

__global__ void shifted_pupil_kernel(cufftComplex* pupil, int h, int w, float dx,
                                     float cutoff, float source_fx, float source_fy) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= h * w) return;
    const int y = index / w;
    const int x = index % w;
    const float fx = (x <= w / 2 ? x : x - w) / (w * dx) + source_fx;
    const float fy = (y <= h / 2 ? y : y - h) / (h * dx) + source_fy;
    pupil[index] = {(fx * fx + fy * fy <= cutoff * cutoff) ? 1.0F : 0.0F, 0.0F};
}

__global__ void multiply_kernel(const cufftComplex* a, const cufftComplex* b,
                                cufftComplex* output, int count, bool conjugate_b) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    cufftComplex rhs = b[index];
    if (conjugate_b) rhs.y = -rhs.y;
    output[index] = complex_mul(a[index], rhs);
}

__global__ void scale_complex_kernel(cufftComplex* values, int count, float scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    values[index].x *= scale;
    values[index].y *= scale;
}

__global__ void accumulate_intensity_kernel(const cufftComplex* field, float* intensity,
                                            int count, float weight) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const cufftComplex value = field[index];
    intensity[index] += weight * (value.x * value.x + value.y * value.y);
}

__global__ void field_gradient_kernel(const cufftComplex* field, const float* intensity_gradient,
                                      cufftComplex* output, int count, float weight) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float factor = 2.0F * weight * intensity_gradient[index];
    output[index] = {factor * field[index].x, factor * field[index].y};
}

__global__ void accumulate_phase_gradient_kernel(const cufftComplex* padded_field,
                                                 const cufftComplex* padded_gradient,
                                                 float* phase_gradient, int h, int w,
                                                 int pw, int oy, int ox) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= h * w) return;
    const int y = index / w;
    const int x = index % w;
    const int padded_index = (y + oy) * pw + x + ox;
    const cufftComplex field = padded_field[padded_index];
    const cufftComplex gradient = padded_gradient[padded_index];
    phase_gradient[index] += gradient.x * (-field.y) + gradient.y * field.x;
}

__global__ void adam_kernel(float* phase, const float* gradient, float* first, float* second,
                            int count, float lr, float beta1, float beta2,
                            float beta1_power, float beta2_power) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float g = gradient[index];
    first[index] = beta1 * first[index] + (1.0F - beta1) * g;
    second[index] = beta2 * second[index] + (1.0F - beta2) * g * g;
    const float m_hat = first[index] / (1.0F - beta1_power);
    const float v_hat = second[index] / (1.0F - beta2_power);
    phase[index] -= lr * m_hat / (sqrtf(v_hat) + 1e-8F);
}

__global__ void add_direction_kernel(float* phase, const float* direction, int count, float step) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) phase[index] += step * direction[index];
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) value += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(value);
}

std::vector<float> annealed_target(const std::vector<float>& base, int h, int w,
                                   int step, int total_steps, float start_sigma, float end_sigma) {
    const float progress = static_cast<float>(step) / std::max(1, total_steps - 1);
    const float sigma = start_sigma * std::pow(end_sigma / start_sigma, progress);
    if (sigma <= 0.3F) return base;
    cv::Mat source(h, w, CV_32F, const_cast<float*>(base.data()));
    cv::Mat blurred;
    const int kernel = static_cast<int>(6.0F * sigma) | 1;
    cv::GaussianBlur(source, blurred, cv::Size(kernel, kernel), sigma);
    return std::vector<float>(blurred.ptr<float>(), blurred.ptr<float>() + h * w);
}

}  // namespace

Config Config::load(const std::string& path) {
    cv::FileStorage storage(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) throw std::runtime_error("cannot open config: " + path);
    Config config;
    const cv::FileNode root = storage.root();
    config.wavelength = read_or<float>(root, "wavelength", config.wavelength);
    config.pixel_size = read_or<float>(root, "pixel_size", config.pixel_size);
    config.numerical_aperture = read_or<float>(root, "NA", config.numerical_aperture);
    config.distance = read_or<float>(root, "distance", config.distance);
    config.learning_rate = read_or<float>(root, "learning_rate", config.learning_rate);
    config.mask_pretrain_steps = read_or<int>(root, "mask_pretrain_steps", config.mask_pretrain_steps);
    config.joint_steps = read_or<int>(root, "joint_steps", config.joint_steps);
    config.joint_version = read_or<int>(root, "joint_optimization_version", config.joint_version);
    config.source_learning_rate = read_or<float>(root, "source_learning_rate", config.source_learning_rate);
    config.cg_learning_rate = read_or<float>(root, "cg_learning_rate", config.cg_learning_rate);
    config.source_cg_learning_rate = read_or<float>(root, "source_cg_learning_rate", config.source_cg_learning_rate);
    config.anneal_start_sigma = read_or<float>(root, "anneal_start_sigma", config.anneal_start_sigma);
    config.anneal_end_sigma = read_or<float>(root, "anneal_end_sigma", config.anneal_end_sigma);
    config.max_source_points = read_or<int>(root, "max_source_points", config.max_source_points);
    config.image_source = read_string_or(root, "image_source", config.image_source);
    config.image_path = read_string_or(root, "image_path", config.image_path);
    config.target_text = read_string_or(root, "target_text", config.target_text);
    config.output = read_string_or(root, "output", config.output);
    const cv::FileNode shape = root["shape"];
    if (!shape.empty() && shape.size() == 2) {
        config.height = static_cast<int>(shape[0]);
        config.width = static_cast<int>(shape[1]);
    }
    const cv::FileNode source = root["source"];
    if (!source.empty()) {
        config.source.sigma_in = read_or<float>(source, "sigma_in", config.source.sigma_in);
        config.source.sigma_out = read_or<float>(source, "sigma_out", config.source.sigma_out);
        config.source.open_angle = read_or<float>(source, "open_angle", config.source.open_angle);
        config.source.offset_angle = read_or<float>(source, "offset_angle", config.source.offset_angle);
        config.source.num_poles = read_or<int>(source, "num_poles", config.source.num_poles);
    }
    return config;
}

struct Optimizer::Impl {
    Config config;
    int h;
    int w;
    int ph;
    int pw;
    int oy;
    int ox;
    int pixels;
    int padded_pixels;
    cufftHandle fft_plan{};
    float* d_phase = nullptr;
    float* d_intensity = nullptr;
    float* d_intensity_gradient = nullptr;
    float* d_phase_gradient = nullptr;
    float* d_adam_first = nullptr;
    float* d_adam_second = nullptr;
    float* d_direction = nullptr;
    cufftComplex* d_padded = nullptr;
    cufftComplex* d_spectrum = nullptr;
    cufftComplex* d_pupil = nullptr;
    cufftComplex* d_temp = nullptr;
    cufftComplex* d_field = nullptr;
    cufftComplex* d_transfer = nullptr;
    std::vector<SourcePoint> source_points;
    int adam_step = 0;

    explicit Impl(Config cfg)
        : config(std::move(cfg)), h(config.height), w(config.width),
          ph(h + 2 * (h / 4)), pw(w + 2 * (w / 4)), oy(h / 4), ox(w / 4),
          pixels(h * w), padded_pixels(ph * pw) {
        CUDA_CHECK(cudaMalloc(&d_phase, pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_intensity, padded_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_intensity_gradient, padded_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_phase_gradient, pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_adam_first, pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_adam_second, pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_direction, pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_padded, padded_pixels * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_spectrum, padded_pixels * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_pupil, padded_pixels * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_temp, padded_pixels * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_field, padded_pixels * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_transfer, padded_pixels * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMemset(d_adam_first, 0, pixels * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_adam_second, 0, pixels * sizeof(float)));
        CUFFT_CHECK(cufftPlan2d(&fft_plan, ph, pw, CUFFT_C2C));

        std::mt19937 generator(42);
        std::normal_distribution<float> distribution(0.0F, 0.01F);
        std::vector<float> initial(pixels);
        std::generate(initial.begin(), initial.end(), [&] { return distribution(generator); });
        CUDA_CHECK(cudaMemcpy(d_phase, initial.data(), pixels * sizeof(float), cudaMemcpyHostToDevice));

        const int blocks = (padded_pixels + 255) / 256;
        transfer_kernel<<<blocks, 256>>>(d_transfer, ph, pw, config.pixel_size,
                                         config.wavelength, config.distance);
        CUDA_CHECK(cudaGetLastError());
        rebuild_source_points();
    }

    ~Impl() {
        if (fft_plan) cufftDestroy(fft_plan);
        cudaFree(d_phase); cudaFree(d_intensity); cudaFree(d_intensity_gradient);
        cudaFree(d_phase_gradient); cudaFree(d_adam_first); cudaFree(d_adam_second);
        cudaFree(d_direction); cudaFree(d_padded); cudaFree(d_spectrum);
        cudaFree(d_pupil); cudaFree(d_temp); cudaFree(d_field); cudaFree(d_transfer);
    }

    void clamp_source() {
        auto& source = config.source;
        source.sigma_in = std::clamp(source.sigma_in, 0.0F, 0.98F);
        source.sigma_out = std::clamp(source.sigma_out, 0.01F, 1.0F);
        source.open_angle = std::clamp(source.open_angle, 1.0F, 360.0F / std::max(1, source.num_poles));
        source.offset_angle = std::fmod(std::max(0.0F, source.offset_angle), 360.0F);
        if (source.sigma_out < source.sigma_in + 0.01F) {
            source.sigma_out = std::min(1.0F, source.sigma_in + 0.01F);
            source.sigma_in = std::max(0.0F, source.sigma_out - 0.01F);
        }
    }

    void rebuild_source_points() {
        clamp_source();
        source_points.clear();
        const float cutoff = config.numerical_aperture / config.wavelength;
        const float period = 2.0F * kPi / std::max(1, config.source.num_poles);
        const float half_angle = 0.5F * config.source.open_angle * kPi / 180.0F;
        const float offset = config.source.offset_angle * kPi / 180.0F;
        for (int y = 0; y < ph; ++y) {
            const float fy = (y <= ph / 2 ? y : y - ph) / (ph * config.pixel_size);
            for (int x = 0; x < pw; ++x) {
                const float fx = (x <= pw / 2 ? x : x - pw) / (pw * config.pixel_size);
                const float rho = std::hypot(fx, fy) / cutoff;
                if (rho < config.source.sigma_in || rho > config.source.sigma_out) continue;
                bool angular_match = true;
                if (config.source.open_angle < 360.0F / std::max(1, config.source.num_poles)) {
                    float local = std::remainder(std::atan2(fy, fx) - offset, period);
                    angular_match = std::fabs(local) <= half_angle;
                }
                if (angular_match) source_points.push_back({fx, fy, 1.0F});
            }
        }
        if (source_points.empty()) throw std::runtime_error("source profile contains no sampled points");
        if (static_cast<int>(source_points.size()) > config.max_source_points) {
            const int stride = static_cast<int>(std::ceil(
                static_cast<float>(source_points.size()) / config.max_source_points));
            std::vector<SourcePoint> sampled;
            for (std::size_t i = 0; i < source_points.size(); i += stride) sampled.push_back(source_points[i]);
            source_points.swap(sampled);
        }
        const float weight = 1.0F / source_points.size();
        for (auto& point : source_points) point.weight = weight;
    }

    void fft(cufftComplex* input, cufftComplex* output, int direction) {
        CUFFT_CHECK(cufftExecC2C(fft_plan, input, output, direction));
    }

    void inverse_and_scale(cufftComplex* input, cufftComplex* output) {
        fft(input, output, CUFFT_INVERSE);
        scale_complex_kernel<<<(padded_pixels + 255) / 256, 256>>>(
            output, padded_pixels, 1.0F / padded_pixels);
    }

    void prepare_mask_spectrum() {
        phase_to_padded_kernel<<<(padded_pixels + 255) / 256, 256>>>(
            d_phase, d_padded, h, w, ph, pw, oy, ox);
        fft(d_padded, d_spectrum, CUFFT_FORWARD);
    }

    void propagate_source(const SourcePoint& point) {
        const int blocks = (padded_pixels + 255) / 256;
        shifted_pupil_kernel<<<blocks, 256>>>(d_pupil, ph, pw, config.pixel_size,
                                              config.numerical_aperture / config.wavelength,
                                              point.fx, point.fy);
        multiply_kernel<<<blocks, 256>>>(d_spectrum, d_pupil, d_temp, padded_pixels, false);
        inverse_and_scale(d_temp, d_field);
        fft(d_field, d_temp, CUFFT_FORWARD);
        multiply_kernel<<<blocks, 256>>>(d_temp, d_transfer, d_field, padded_pixels, false);
        inverse_and_scale(d_field, d_temp);
        std::swap(d_field, d_temp);
    }

    std::vector<float> forward_padded() {
        CUDA_CHECK(cudaMemset(d_intensity, 0, padded_pixels * sizeof(float)));
        prepare_mask_spectrum();
        for (const auto& point : source_points) {
            propagate_source(point);
            accumulate_intensity_kernel<<<(padded_pixels + 255) / 256, 256>>>(
                d_field, d_intensity, padded_pixels, point.weight);
        }
        std::vector<float> result(padded_pixels);
        CUDA_CHECK(cudaMemcpy(result.data(), d_intensity, padded_pixels * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    }

    float loss_and_intensity_gradient(const std::vector<float>& target, bool upload_gradient) {
        const std::vector<float> intensity = forward_padded();
        float maximum = 1e-12F;
        int maximum_index = 0;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const int index = (y + oy) * pw + x + ox;
            if (intensity[index] > maximum) { maximum = intensity[index]; maximum_index = index; }
        }
        std::vector<float> gradient(padded_pixels, 0.0F);
        double loss = 0.0;
        double maximum_derivative = 0.0;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const int local = y * w + x;
            const int index = (y + oy) * pw + x + ox;
            const float normalized = intensity[index] / maximum;
            const float difference = normalized - target[local];
            loss += static_cast<double>(difference) * difference / pixels;
            const float derivative_normalized = 2.0F * difference / pixels;
            gradient[index] = derivative_normalized / maximum;
            maximum_derivative -= derivative_normalized * intensity[index] / (maximum * maximum);
        }
        gradient[maximum_index] += static_cast<float>(maximum_derivative);
        if (upload_gradient) {
            CUDA_CHECK(cudaMemcpy(d_intensity_gradient, gradient.data(), padded_pixels * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        return static_cast<float>(loss);
    }

    std::vector<float> phase_gradient(const std::vector<float>& target, float& loss) {
        loss = loss_and_intensity_gradient(target, true);
        CUDA_CHECK(cudaMemset(d_phase_gradient, 0, pixels * sizeof(float)));
        prepare_mask_spectrum();
        const int padded_blocks = (padded_pixels + 255) / 256;
        for (const auto& point : source_points) {
            propagate_source(point);
            field_gradient_kernel<<<padded_blocks, 256>>>(
                d_field, d_intensity_gradient, d_temp, padded_pixels, point.weight);
            fft(d_temp, d_field, CUFFT_FORWARD);
            multiply_kernel<<<padded_blocks, 256>>>(d_field, d_transfer, d_temp, padded_pixels, true);
            inverse_and_scale(d_temp, d_field);
            fft(d_field, d_temp, CUFFT_FORWARD);
            multiply_kernel<<<padded_blocks, 256>>>(d_temp, d_pupil, d_field, padded_pixels, true);
            inverse_and_scale(d_field, d_temp);
            accumulate_phase_gradient_kernel<<<(pixels + 255) / 256, 256>>>(
                d_padded, d_temp, d_phase_gradient, h, w, pw, oy, ox);
        }
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> gradient(pixels);
        CUDA_CHECK(cudaMemcpy(gradient.data(), d_phase_gradient, pixels * sizeof(float), cudaMemcpyDeviceToHost));
        return gradient;
    }

    void adam_update() {
        ++adam_step;
        adam_kernel<<<(pixels + 255) / 256, 256>>>(
            d_phase, d_phase_gradient, d_adam_first, d_adam_second, pixels,
            config.learning_rate, 0.9F, 0.999F, std::pow(0.9F, adam_step), std::pow(0.999F, adam_step));
    }

    std::vector<float> source_gradient(const std::vector<float>& target) {
        const SourceParameters original = config.source;
        std::vector<float> gradient(3, 0.0F);
        const float eps[3] = {0.005F, 0.005F, 0.5F};
        for (int parameter = 0; parameter < 3; ++parameter) {
            auto set_value = [&](float delta) {
                config.source = original;
                if (parameter == 0) config.source.sigma_in += delta;
                if (parameter == 1) config.source.sigma_out += delta;
                if (parameter == 2) config.source.open_angle += delta;
                rebuild_source_points();
            };
            set_value(eps[parameter]);
            const float plus = loss_and_intensity_gradient(target, false);
            const float plus_value = parameter == 0 ? config.source.sigma_in
                                   : parameter == 1 ? config.source.sigma_out : config.source.open_angle;
            set_value(-eps[parameter]);
            const float minus = loss_and_intensity_gradient(target, false);
            const float minus_value = parameter == 0 ? config.source.sigma_in
                                    : parameter == 1 ? config.source.sigma_out : config.source.open_angle;
            gradient[parameter] = (plus - minus) / std::max(plus_value - minus_value, 1e-12F);
        }
        config.source = original;
        rebuild_source_points();
        return gradient;
    }

    void update_source(const std::vector<float>& direction, float step) {
        config.source.sigma_in += step * direction[0];
        config.source.sigma_out += step * direction[1];
        config.source.open_angle += step * direction[2];
        rebuild_source_points();
    }

    void run(const std::vector<float>& base_target) {
        const int total = config.mask_pretrain_steps + config.joint_steps;
        for (int step = 0; step < config.mask_pretrain_steps; ++step) {
            const auto target = annealed_target(base_target, h, w, step, total,
                                                config.anneal_start_sigma, config.anneal_end_sigma);
            float loss = 0.0F;
            phase_gradient(target, loss);
            adam_update();
            if (step % 20 == 0) std::cout << "Mask " << step << '/' << total << " loss=" << loss << '\n';
        }

        if (config.joint_version == 1) {
            for (int joint = 0; joint < config.joint_steps; ++joint) {
                const int step = config.mask_pretrain_steps + joint;
                const auto target = annealed_target(base_target, h, w, step, total,
                                                    config.anneal_start_sigma, config.anneal_end_sigma);
                float loss = 0.0F;
                phase_gradient(target, loss);
                adam_update();
                const auto source_grad = source_gradient(target);
                std::vector<float> descent(source_grad.size());
                std::transform(source_grad.begin(), source_grad.end(), descent.begin(), std::negate<float>());
                update_source(descent, config.source_learning_rate);
                if (step % 10 == 0) std::cout << "Joint " << step << '/' << total << " loss=" << loss << '\n';
            }
            return;
        }

        std::vector<float> previous_phase_gradient;
        std::vector<float> previous_source_gradient;
        std::vector<float> phase_direction;
        std::vector<float> source_direction;
        for (int joint = 0; joint < config.joint_steps; ++joint) {
            const int step = config.mask_pretrain_steps + joint;
            const auto target = annealed_target(base_target, h, w, step, total,
                                                config.anneal_start_sigma, config.anneal_end_sigma);
            float loss = 0.0F;
            const auto phase_grad = phase_gradient(target, loss);
            const auto source_grad = source_gradient(target);
            float beta = 0.0F;
            if (!previous_phase_gradient.empty()) {
                std::vector<float> phase_delta(pixels);
                std::vector<float> source_delta(3);
                for (int i = 0; i < pixels; ++i) phase_delta[i] = phase_grad[i] - previous_phase_gradient[i];
                for (int i = 0; i < 3; ++i) source_delta[i] = source_grad[i] - previous_source_gradient[i];
                const float numerator = dot(phase_grad, phase_delta) + dot(source_grad, source_delta);
                const float denominator = dot(previous_phase_gradient, previous_phase_gradient)
                                        + dot(previous_source_gradient, previous_source_gradient);
                beta = std::max(0.0F, numerator / std::max(denominator, 1e-20F));
            }
            if (phase_direction.empty()) {
                phase_direction.resize(pixels);
                source_direction.resize(3);
            }
            for (int i = 0; i < pixels; ++i) phase_direction[i] = -phase_grad[i] + beta * phase_direction[i];
            for (int i = 0; i < 3; ++i) source_direction[i] = -source_grad[i] + beta * source_direction[i];
            CUDA_CHECK(cudaMemcpy(d_direction, phase_direction.data(), pixels * sizeof(float), cudaMemcpyHostToDevice));
            add_direction_kernel<<<(pixels + 255) / 256, 256>>>(
                d_phase, d_direction, pixels, config.cg_learning_rate);
            update_source(source_direction, config.source_cg_learning_rate);
            previous_phase_gradient = phase_grad;
            previous_source_gradient = source_grad;
            if (step % 10 == 0) std::cout << "CG joint " << step << '/' << total << " loss=" << loss << '\n';
        }
    }
};

Optimizer::Optimizer(Config config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Optimizer::~Optimizer() = default;
void Optimizer::run(const std::vector<float>& target) { impl_->run(target); }

std::vector<float> Optimizer::final_intensity() {
    const auto padded = impl_->forward_padded();
    std::vector<float> cropped(impl_->pixels);
    for (int y = 0; y < impl_->h; ++y) for (int x = 0; x < impl_->w; ++x) {
        cropped[y * impl_->w + x] = padded[(y + impl_->oy) * impl_->pw + x + impl_->ox];
    }
    return cropped;
}

std::vector<float> Optimizer::phase() const {
    std::vector<float> result(impl_->pixels);
    CUDA_CHECK(cudaMemcpy(result.data(), impl_->d_phase, impl_->pixels * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

const SourceParameters& Optimizer::source_parameters() const { return impl_->config.source; }

std::vector<unsigned char> Optimizer::source_profile() const {
    std::vector<unsigned char> profile(impl_->padded_pixels, 0);
    const float cutoff = impl_->config.numerical_aperture / impl_->config.wavelength;
    for (const auto& point : impl_->source_points) {
        int x = static_cast<int>(std::lround(point.fx * impl_->pw * impl_->config.pixel_size));
        int y = static_cast<int>(std::lround(point.fy * impl_->ph * impl_->config.pixel_size));
        x = (x % impl_->pw + impl_->pw) % impl_->pw;
        y = (y % impl_->ph + impl_->ph) % impl_->ph;
        (void)cutoff;
        const int shifted_x = (x + impl_->pw / 2) % impl_->pw;
        const int shifted_y = (y + impl_->ph / 2) % impl_->ph;
        profile[shifted_y * impl_->pw + shifted_x] = 255;
    }
    return profile;
}

std::vector<float> load_target(const Config& config, const std::string& config_dir) {
    cv::Mat image;
    if (config.image_source == "file") {
        std::filesystem::path path = config.image_path;
        if (path.is_relative()) path = std::filesystem::path(config_dir) / path;
        image = cv::imread(path.lexically_normal().string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) throw std::runtime_error("cannot load target image: " + path.string());
        cv::resize(image, image, cv::Size(config.width, config.height));
    } else if (config.image_source == "text") {
        image = cv::Mat::zeros(config.height, config.width, CV_8U);
        int baseline = 0;
        const cv::Size size = cv::getTextSize(config.target_text, cv::FONT_HERSHEY_SIMPLEX, 1.5, 3, &baseline);
        const cv::Point origin((config.width - size.width) / 2, (config.height + size.height) / 2);
        cv::putText(image, config.target_text, origin, cv::FONT_HERSHEY_SIMPLEX, 1.5, 255, 3, cv::LINE_AA);
    } else {
        throw std::runtime_error("unsupported image_source: " + config.image_source);
    }
    cv::Mat floating;
    image.convertTo(floating, CV_32F, 1.0 / 255.0);
    return std::vector<float>(floating.ptr<float>(), floating.ptr<float>() + config.height * config.width);
}

void write_results(Optimizer& optimizer, const Config& config, const std::string& output_dir) {
    std::filesystem::create_directories(output_dir);
    auto intensity = optimizer.final_intensity();
    const float maximum = std::max(*std::max_element(intensity.begin(), intensity.end()), 1e-12F);
    cv::Mat reconstructed(config.height, config.width, CV_8U);
    for (int i = 0; i < config.height * config.width; ++i) {
        reconstructed.data[i] = static_cast<unsigned char>(std::clamp(intensity[i] / maximum * 255.0F, 0.0F, 255.0F));
    }
    cv::imwrite((std::filesystem::path(output_dir) / "reconstructed_image_cpp.png").string(), reconstructed);

    const auto phase = optimizer.phase();
    cv::Mat phase_image(config.height, config.width, CV_8U);
    for (int i = 0; i < config.height * config.width; ++i) {
        float wrapped = std::fmod(phase[i], 2.0F * kPi);
        if (wrapped < 0.0F) wrapped += 2.0F * kPi;
        phase_image.data[i] = static_cast<unsigned char>(wrapped / (2.0F * kPi) * 255.0F);
    }
    cv::imwrite((std::filesystem::path(output_dir) / "optimized_doe_phase_cpp.png").string(), phase_image);

    const auto source = optimizer.source_profile();
    cv::Mat source_image(config.height + 2 * (config.height / 4),
                         config.width + 2 * (config.width / 4), CV_8U,
                         const_cast<unsigned char*>(source.data()));
    cv::imwrite((std::filesystem::path(output_dir) / "source_profile_cpp.png").string(), source_image);

    const auto& parameters = optimizer.source_parameters();
    std::ofstream json(std::filesystem::path(output_dir) / "optimized_source_params_cpp.json");
    json << std::setprecision(8) << "{\n"
         << "  \"sigma_in\": " << parameters.sigma_in << ",\n"
         << "  \"sigma_out\": " << parameters.sigma_out << ",\n"
         << "  \"open_angle\": " << parameters.open_angle << ",\n"
         << "  \"offset_angle\": " << parameters.offset_angle << "\n}\n";
}

}  // namespace diffrax
