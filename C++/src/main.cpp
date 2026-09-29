#include "diffrax/optimizer.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    try {
        const fs::path config_path = argc > 1 ? fs::path(argv[1]) : fs::path("../Python/config.json");
        const fs::path absolute_config = fs::absolute(config_path).lexically_normal();
        diffrax::Config config = diffrax::Config::load(absolute_config.string());

        fs::path output = config.output;
        if (output.is_relative()) {
            output = absolute_config.parent_path() / output;
        }
        output = output.lexically_normal();
        fs::create_directories(output);

        std::cout << "DIFFRA-X CUDA optimizer\n"
                  << "  grid: " << config.width << " x " << config.height << '\n'
                  << "  joint optimization: version " << config.joint_version << '\n'
                  << "  output: " << output << std::endl;

        auto target = diffrax::load_target(config, absolute_config.parent_path().string());
        diffrax::Optimizer optimizer(config);
        optimizer.run(target);
        diffrax::write_results(optimizer, config, output.string());
        std::cout << "Optimization complete." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "diffrax: " << error.what() << std::endl;
        return 1;
    }
}
