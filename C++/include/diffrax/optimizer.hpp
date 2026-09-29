#pragma once

#include <memory>
#include <string>
#include <vector>

namespace diffrax {

struct SourceParameters {
    float sigma_in = 0.0F;
    float sigma_out = 0.2F;
    float open_angle = 35.0F;
    float offset_angle = 0.0F;
    int num_poles = 2;
};

struct Config {
    int height = 256;
    int width = 256;
    float wavelength = 532e-9F;
    float pixel_size = 0.5e-6F;
    float numerical_aperture = 0.6F;
    float distance = 1e-3F;
    float learning_rate = 0.05F;
    int mask_pretrain_steps = 200;
    int joint_steps = 50;
    int joint_version = 1;
    float source_learning_rate = 0.02F;
    float cg_learning_rate = 0.05F;
    float source_cg_learning_rate = 0.02F;
    float anneal_start_sigma = 10.0F;
    float anneal_end_sigma = 1e-3F;
    int max_source_points = 256;
    std::string image_source = "file";
    std::string image_path;
    std::string target_text = "ODE";
    std::string output = "./output";
    SourceParameters source;

    static Config load(const std::string& path);
};

class Optimizer {
public:
    explicit Optimizer(Config config);
    ~Optimizer();
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;

    void run(const std::vector<float>& target);
    std::vector<float> final_intensity();
    std::vector<float> phase() const;
    std::vector<unsigned char> source_profile() const;
    const SourceParameters& source_parameters() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<float> load_target(const Config& config, const std::string& config_dir);
void write_results(Optimizer& optimizer, const Config& config, const std::string& output_dir);

}  // namespace diffrax
