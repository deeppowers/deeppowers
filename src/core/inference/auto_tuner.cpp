#include "auto_tuner.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace deeppowers {

AutoTuner::AutoTuner(const TuningConfig& config) 
    : config_(config), 
      random_generator_(std::random_device{}()) {
}

AutoTuner::~AutoTuner() = default;

TuningResult AutoTuner::tune(Model* model, const std::vector<std::vector<int>>& sample_inputs) {
    TuningResult result;
    result.trials_completed = 0;
    
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        std::unordered_map<std::string, float> best_params;
        
        switch (config_.method) {
            case TuningMethod::GRID_SEARCH:
                best_params = run_grid_search(model, sample_inputs);
                break;
            case TuningMethod::RANDOM_SEARCH:
                best_params = run_random_search(model, sample_inputs);
                break;
            case TuningMethod::BAYESIAN_OPT:
                best_params = run_bayesian_optimization(model, sample_inputs);
                break;
            case TuningMethod::GENETIC_ALGORITHM:
                best_params = run_genetic_algorithm(model, sample_inputs);
                break;
            case TuningMethod::ANNEALING:
                best_params = run_simulated_annealing(model, sample_inputs);
                break;
        }
        
        result.success = true;
        result.best_params = best_params;
        result.all_trials = all_trials_;
        result.trials_completed = all_trials_.size();
        
        // Perform final benchmark with best parameters
        auto metrics = benchmark_model(model, best_params, sample_inputs);
        result.latency_ms = metrics["latency"];
        result.throughput = metrics["throughput"];
        result.memory_mb = metrics["memory"];
        result.accuracy = metrics["accuracy"];
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.tuning_time_seconds = std::chrono::duration<float>(end_time - start_time).count();
    
    return result;
}

bool AutoTuner::apply_best_parameters(Model* model, const std::unordered_map<std::string, float>& params) {
    try {
        if (!model) {
            throw std::runtime_error("Invalid model pointer");
        }
        
        for (const auto& param : config_.parameters) {
            if (params.find(param.name) == params.end()) {
                throw std::runtime_error("Missing parameter: " + param.name);
            }
        }
        
        for (const auto& [param_name, value] : params) {
            // 
            auto it = std::find_if(config_.parameters.begin(), config_.parameters.end(),
                [&param_name](const TuningParameter& p) { return p.name == param_name; });
            
            if (it == config_.parameters.end()) {
                if (config_.verbose) {
                    std::cerr << "Warning: Unknown parameter " << param_name << std::endl;
                }
                continue;
            }
            
            // 
            const auto& param = *it;
            if (param.is_discrete) {
                // ，
                auto value_it = std::find(param.discrete_values.begin(), param.discrete_values.end(), value);
                if (value_it == param.discrete_values.end()) {
                    throw std::runtime_error("Invalid discrete value for parameter " + param_name);
                }
            } else {
                // ，
                if (value < param.min_value || value > param.max_value) {
                    throw std::runtime_error("Value out of range for parameter " + param_name);
                }
            }
            
            // 
            if (param_name.find("batch_size") != std::string::npos) {
                model->set_batch_size(static_cast<int>(value));
            }
            else if (param_name.find("thread_count") != std::string::npos) {
                model->set_thread_count(static_cast<int>(value));
            }
            else if (param_name.find("cache_size") != std::string::npos) {
                model->set_cache_size(static_cast<size_t>(value));
            }
            else if (param_name.find("precision") != std::string::npos) {
                model->set_precision(static_cast<int>(value));
            }
            else if (param_name.find("optimization_level") != std::string::npos) {
                model->set_optimization_level(static_cast<int>(value));
            }
            else if (param_name.find("memory_limit") != std::string::npos) {
                model->set_memory_limit(static_cast<size_t>(value));
            }
            else if (param_name.find("compute_type") != std::string::npos) {
                model->set_compute_type(static_cast<int>(value));
            }
            else {
                // ，
                model->set_parameter(param_name, value);
            }
        }
        
        // 
        if (!model->validate_parameters()) {
            throw std::runtime_error("Parameter validation failed");
        }
        
        return true;
    } catch (const std::exception& e) {
        if (config_.verbose) {
            std::cerr << "Failed to apply parameters: " << e.what() << std::endl;
        }
        return false;
    }
}

void AutoTuner::set_config(const TuningConfig& config) {
    config_ = config;
}

TuningConfig AutoTuner::get_config() const {
    return config_;
}

void AutoTuner::add_parameter(const TuningParameter& param) {
    config_.parameters.push_back(param);
}

void AutoTuner::set_objective(TuningObjective objective) {
    config_.objective = objective;
}

void AutoTuner::set_method(TuningMethod method) {
    config_.method = method;
}

void AutoTuner::register_custom_evaluator(
    std::function<float(Model*, const std::unordered_map<std::string, float>&)> eval_func) {
    custom_evaluator_ = std::move(eval_func);
}

float AutoTuner::evaluate_parameters(
    Model* model,
    const std::unordered_map<std::string, float>& params,
    const std::vector<std::vector<int>>& sample_inputs) {
    
    if (custom_evaluator_) {
        return custom_evaluator_(model, params);
    }
    
    auto metrics = benchmark_model(model, params, sample_inputs);
    return calculate_objective_score(
        metrics["latency"],
        metrics["throughput"],
        metrics["memory"],
        metrics["accuracy"]
    );
}

std::unordered_map<std::string, float> AutoTuner::generate_random_parameters() {
    std::unordered_map<std::string, float> params;
    
    for (const auto& param : config_.parameters) {
        if (param.is_discrete) {
            std::uniform_int_distribution<size_t> dist(0, param.discrete_values.size() - 1);
            params[param.name] = param.discrete_values[dist(random_generator_)];
        } else {
            float value;
            if (param.is_log_scale) {
                std::uniform_real_distribution<float> dist(
                    std::log(param.min_value),
                    std::log(param.max_value)
                );
                value = std::exp(dist(random_generator_));
            } else {
                std::uniform_real_distribution<float> dist(param.min_value, param.max_value);
                value = dist(random_generator_);
            }
            params[param.name] = value;
        }
    }
    
    return params;
}

float AutoTuner::calculate_objective_score(
    float latency,
    float throughput,
    float memory,
    float accuracy) {
    
    switch (config_.objective) {
        case TuningObjective::LATENCY:
            return -latency;  // Negative because we want to minimize latency
        case TuningObjective::THROUGHPUT:
            return throughput;
        case TuningObjective::MEMORY:
            return -memory;  // Negative because we want to minimize memory usage
        case TuningObjective::ACCURACY:
            return accuracy;
        case TuningObjective::BALANCED:
            // Normalized weighted sum of all metrics
            return (-0.3f * latency / 100.0f) +  // Assuming typical latency is 0-100ms
                   (0.3f * throughput / 1000.0f) +  // Assuming typical throughput is 0-1000 tokens/s
                   (-0.2f * memory / 1024.0f) +  // Assuming typical memory is 0-1024MB
                   (0.2f * accuracy / 100.0f);  // Accuracy is already 0-100
    }
    
    return 0.0f;  // Should never reach here
}

// Implementation of search algorithms
std::unordered_map<std::string, float> AutoTuner::run_grid_search(
    Model* model,
    const std::vector<std::vector<int>>& sample_inputs) {
    
    std::unordered_map<std::string, float> best_params;
    float best_score = -std::numeric_limits<float>::max();
    
    // 
    std::vector<std::vector<float>> parameter_grids;
    std::vector<std::string> parameter_names;
    
    for (const auto& param : config_.parameters) {
        parameter_names.push_back(param.name);
        
        if (param.is_discrete) {
            parameter_grids.push_back(param.discrete_values);
        } else {
            std::vector<float> grid;
            int num_points = 5;  // 
            
            if (param.is_log_scale) {
                float log_min = std::log(param.min_value);
                float log_max = std::log(param.max_value);
                float step = (log_max - log_min) / (num_points - 1);
                
                for (int i = 0; i < num_points; ++i) {
                    grid.push_back(std::exp(log_min + i * step));
                }
            } else {
                float step = (param.max_value - param.min_value) / (num_points - 1);
                for (int i = 0; i < num_points; ++i) {
                    grid.push_back(param.min_value + i * step);
                }
            }
            parameter_grids.push_back(grid);
        }
    }
    
    // ，
    std::function<void(size_t, std::unordered_map<std::string, float>&)> try_combinations =
        [&](size_t param_index, std::unordered_map<std::string, float>& current_params) {
            if (param_index == parameter_names.size()) {
                // 
                float score = evaluate_parameters(model, current_params, sample_inputs);
                all_trials_.push_back(current_params);
                
                if (score > best_score) {
                    best_score = score;
                    best_params = current_params;
                }
                
                if (config_.verbose) {
                    std::cout << "Grid point evaluated - Score: " << score << std::endl;
                }
                return;
            }
            
            // 
            const std::string& param_name = parameter_names[param_index];
            const std::vector<float>& grid = parameter_grids[param_index];
            
            for (float value : grid) {
                current_params[param_name] = value;
                try_combinations(param_index + 1, current_params);
            }
        };
    
    // 
    std::unordered_map<std::string, float> current_params;
    try_combinations(0, current_params);
    
    return best_params;
}

std::unordered_map<std::string, float> AutoTuner::run_random_search(
    Model* model,
    const std::vector<std::vector<int>>& sample_inputs) {
    
    std::unordered_map<std::string, float> best_params;
    float best_score = -std::numeric_limits<float>::max();
    
    for (int trial = 0; trial < config_.max_trials; ++trial) {
        auto params = generate_random_parameters();
        float score = evaluate_parameters(model, params, sample_inputs);
        
        all_trials_.push_back(params);
        
        if (score > best_score) {
            best_score = score;
            best_params = params;
        }
        
        if (config_.verbose) {
            std::cout << "Trial " << trial + 1 << "/" << config_.max_trials
                     << " - Score: " << score << std::endl;
        }
    }
    
    return best_params;
}

std::unordered_map<std::string, float> AutoTuner::run_bayesian_optimization(
    Model* model,
    const std::vector<std::vector<int>>& sample_inputs) {
    
    std::unordered_map<std::string, float> best_params;
    float best_score = -std::numeric_limits<float>::max();
    
    // 
    const float length_scale = 1.0f;
    const float signal_variance = 1.0f;
    const float noise_variance = 0.1f;
    
    // 
    std::vector<std::unordered_map<std::string, float>> X;  // 
    std::vector<float> y;  // 
    
    // 
    const int n_initial = 5;
    for (int i = 0; i < n_initial; ++i) {
        auto params = generate_random_parameters();
        float score = evaluate_parameters(model, params, sample_inputs);
        
        X.push_back(params);
        y.push_back(score);
        all_trials_.push_back(params);
        
        if (score > best_score) {
            best_score = score;
            best_params = params;
        }
    }
    
    // 
    for (int iter = n_initial; iter < config_.max_trials; ++iter) {
        //  K
        std::vector<std::vector<float>> K(X.size(), std::vector<float>(X.size()));
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j <= i; ++j) {
                float dist = 0.0f;
                for (const auto& param : config_.parameters) {
                    float diff = X[i][param.name] - X[j][param.name];
                    if (param.is_discrete) {
                        dist += (diff != 0) ? 1.0f : 0.0f;
                    } else {
                        if (param.is_log_scale) {
                            diff = std::log(X[i][param.name]) - std::log(X[j][param.name]);
                        }
                        dist += diff * diff;
                    }
                }
                
                K[i][j] = K[j][i] = signal_variance * std::exp(-0.5f * dist / (length_scale * length_scale));
                if (i == j) {
                    K[i][i] += noise_variance;
                }
            }
        }
        
        // 
        std::unordered_map<std::string, float> next_params;
        float max_acq = -std::numeric_limits<float>::max();
        
        const int n_candidates = 100;
        for (int c = 0; c < n_candidates; ++c) {
            auto candidate = generate_random_parameters();
            
            // 
            std::vector<float> k_star(X.size());
            for (size_t i = 0; i < X.size(); ++i) {
                float dist = 0.0f;
                for (const auto& param : config_.parameters) {
                    float diff = candidate[param.name] - X[i][param.name];
                    if (param.is_discrete) {
                        dist += (diff != 0) ? 1.0f : 0.0f;
                    } else {
                        if (param.is_log_scale) {
                            diff = std::log(candidate[param.name]) - std::log(X[i][param.name]);
                        }
                        dist += diff * diff;
                    }
                }
                k_star[i] = signal_variance * std::exp(-0.5f * dist / (length_scale * length_scale));
            }
            
            // （）
            float mean = 0.0f;
            float var = signal_variance;
            
            for (size_t i = 0; i < X.size(); ++i) {
                float weight = 0.0f;
                for (size_t j = 0; j < X.size(); ++j) {
                    weight += k_star[j] / K[i][j];
                }
                mean += weight * y[i];
                var -= k_star[i] * weight;
            }
            
            // （）
            const float beta = 2.0f;
            float acq = mean + beta * std::sqrt(std::max(0.0f, var));
            
            if (acq > max_acq) {
                max_acq = acq;
                next_params = candidate;
            }
        }
        
        // 
        float score = evaluate_parameters(model, next_params, sample_inputs);
        X.push_back(next_params);
        y.push_back(score);
        all_trials_.push_back(next_params);
        
        if (score > best_score) {
            best_score = score;
            best_params = next_params;
        }
        
        if (config_.verbose) {
            std::cout << "Bayesian optimization iteration " << iter + 1 << "/" << config_.max_trials
                     << " - Score: " << score << " (best: " << best_score << ")" << std::endl;
        }
    }
    
    return best_params;
}

std::unordered_map<std::string, float> AutoTuner::run_genetic_algorithm(
    Model* model,
    const std::vector<std::vector<int>>& sample_inputs) {
    
    const int population_size = 20;
    const float mutation_rate = 0.1f;
    const float crossover_rate = 0.8f;
    
    // 
    std::vector<std::unordered_map<std::string, float>> population;
    std::vector<float> fitness;
    std::unordered_map<std::string, float> best_params;
    float best_score = -std::numeric_limits<float>::max();
    
    // 
    for (int i = 0; i < population_size; ++i) {
        auto params = generate_random_parameters();
        float score = evaluate_parameters(model, params, sample_inputs);
        
        population.push_back(params);
        fitness.push_back(score);
        all_trials_.push_back(params);
        
        if (score > best_score) {
            best_score = score;
            best_params = params;
        }
    }
    
    // 
    int generation = 1;
    int stall_generations = 0;
    const int max_stall_generations = 5;
    
    while (generation < config_.max_trials && stall_generations < max_stall_generations) {
        // ：
        std::vector<float> selection_prob(population_size);
        float total_fitness = 0.0f;
        float min_fitness = *std::min_element(fitness.begin(), fitness.end());
        
        // 
        for (int i = 0; i < population_size; ++i) {
            selection_prob[i] = fitness[i] - min_fitness + 1e-6f;
            total_fitness += selection_prob[i];
        }
        
        // 
        for (int i = 0; i < population_size; ++i) {
            selection_prob[i] /= total_fitness;
            if (i > 0) {
                selection_prob[i] += selection_prob[i-1];
            }
        }
        
        // 
        std::vector<std::unordered_map<std::string, float>> new_population;
        std::vector<float> new_fitness;
        
        while (new_population.size() < population_size) {
            // 
            auto select_parent = [&]() -> const std::unordered_map<std::string, float>& {
                float r = std::uniform_real_distribution<float>(0, 1)(random_generator_);
                auto it = std::lower_bound(selection_prob.begin(), selection_prob.end(), r);
                int idx = std::distance(selection_prob.begin(), it);
                return population[idx];
            };
            
            const auto& parent1 = select_parent();
            const auto& parent2 = select_parent();
            
            // 
            std::unordered_map<std::string, float> child = parent1;
            if (std::uniform_real_distribution<float>(0, 1)(random_generator_) < crossover_rate) {
                for (const auto& param : config_.parameters) {
                    if (std::uniform_real_distribution<float>(0, 1)(random_generator_) < 0.5f) {
                        child[param.name] = parent2.at(param.name);
                    }
                }
            }
            
            // 
            for (const auto& param : config_.parameters) {
                if (std::uniform_real_distribution<float>(0, 1)(random_generator_) < mutation_rate) {
                    if (param.is_discrete) {
                        std::uniform_int_distribution<size_t> dist(0, param.discrete_values.size() - 1);
                        child[param.name] = param.discrete_values[dist(random_generator_)];
                    } else {
                        float range = param.max_value - param.min_value;
                        float mutation = std::normal_distribution<float>(0, range * 0.1f)(random_generator_);
                        child[param.name] = std::clamp(
                            child[param.name] + mutation,
                            param.min_value,
                            param.max_value
                        );
                    }
                }
            }
            
            // 
            float score = evaluate_parameters(model, child, sample_inputs);
            new_population.push_back(child);
            new_fitness.push_back(score);
            all_trials_.push_back(child);
            
            if (score > best_score) {
                best_score = score;
                best_params = child;
                stall_generations = 0;
            }
        }
        
        // 
        population = std::move(new_population);
        fitness = std::move(new_fitness);
        
        if (config_.verbose) {
            std::cout << "Generation " << generation << " - Best score: " << best_score << std::endl;
        }
        
        ++generation;
        ++stall_generations;
    }
    
    return best_params;
}

std::unordered_map<std::string, float> AutoTuner::run_simulated_annealing(
    Model* model,
    const std::vector<std::vector<int>>& sample_inputs) {
    
    // 
    const float initial_temp = 1.0f;
    const float cooling_rate = 0.95f;
    const int steps_per_temp = 5;
    
    // 
    auto current_params = generate_random_parameters();
    float current_score = evaluate_parameters(model, current_params, sample_inputs);
    all_trials_.push_back(current_params);
    
    auto best_params = current_params;
    float best_score = current_score;
    
    // 
    float temperature = initial_temp;
    int iteration = 1;
    
    while (iteration < config_.max_trials) {
        for (int step = 0; step < steps_per_temp && iteration < config_.max_trials; ++step, ++iteration) {
            // 
            auto neighbor_params = current_params;
            
            // 
            const auto& param = config_.parameters[
                std::uniform_int_distribution<size_t>(0, config_.parameters.size() - 1)(random_generator_)
            ];
            
            if (param.is_discrete) {
                // ，
                std::vector<float> available_values = param.discrete_values;
                auto it = std::find(available_values.begin(), available_values.end(), current_params[param.name]);
                if (it != available_values.end()) {
                    available_values.erase(it);
                }
                if (!available_values.empty()) {
                    std::uniform_int_distribution<size_t> dist(0, available_values.size() - 1);
                    neighbor_params[param.name] = available_values[dist(random_generator_)];
                }
            } else {
                // ，
                float range = param.max_value - param.min_value;
                float noise = std::normal_distribution<float>(0, range * temperature * 0.1f)(random_generator_);
                
                if (param.is_log_scale) {
                    float log_value = std::log(current_params[param.name]);
                    float log_noise = noise / current_params[param.name];  // 
                    neighbor_params[param.name] = std::exp(std::clamp(
                        log_value + log_noise,
                        std::log(param.min_value),
                        std::log(param.max_value)
                    ));
                } else {
                    neighbor_params[param.name] = std::clamp(
                        current_params[param.name] + noise,
                        param.min_value,
                        param.max_value
                    );
                }
            }
            
            // 
            float neighbor_score = evaluate_parameters(model, neighbor_params, sample_inputs);
            all_trials_.push_back(neighbor_params);
            
            // 
            float delta = neighbor_score - current_score;
            float acceptance_prob = delta > 0 ? 1.0f : std::exp(delta / temperature);
            
            // 
            if (std::uniform_real_distribution<float>(0, 1)(random_generator_) < acceptance_prob) {
                current_params = neighbor_params;
                current_score = neighbor_score;
                
                // 
                if (current_score > best_score) {
                    best_score = current_score;
                    best_params = current_params;
                }
            }
            
            if (config_.verbose) {
                std::cout << "Iteration " << iteration << "/" << config_.max_trials
                         << " - Temperature: " << temperature
                         << " - Current score: " << current_score
                         << " - Best score: " << best_score << std::endl;
            }
        }
        
        // 
        temperature *= cooling_rate;
        
        // ，
        if (temperature < 1e-4f) {
            temperature = initial_temp;
        }
    }
    
    return best_params;
}

void AutoTuner::set_shape_configs(const std::vector<ShapeConfig>& configs) {
    config_.shape_configs = configs;
}

void AutoTuner::set_shape_strategy(DynamicShapeStrategy strategy) {
    config_.shape_strategy = strategy;
}

bool AutoTuner::validate_shape(const std::vector<int>& shape, const ShapeConfig& config) const {
    // 
    if (shape.size() != config.dimensions.size()) {
        return false;
    }
    
    // 
    for (size_t i = 0; i < shape.size(); ++i) {
        const auto& dim = config.dimensions[i];
        int size = shape[i];
        
        if (dim.is_dynamic) {
            // ，
            if (size < dim.min_size || size > dim.max_size) {
                return false;
            }
        } else {
            // ，
            if (std::find(dim.fixed_sizes.begin(), dim.fixed_sizes.end(), size) == dim.fixed_sizes.end()) {
                return false;
            }
        }
    }
    
    return true;
}

std::vector<int> AutoTuner::generate_random_shape(const ShapeConfig& config) const {
    std::vector<int> shape;
    shape.reserve(config.dimensions.size());
    
    for (const auto& dim : config.dimensions) {
        if (dim.is_dynamic) {
            // ，
            std::uniform_int_distribution<int> dist(dim.min_size, dim.max_size);
            shape.push_back(dist(random_generator_));
        } else {
            // ，
            if (!dim.fixed_sizes.empty()) {
                std::uniform_int_distribution<size_t> dist(0, dim.fixed_sizes.size() - 1);
                shape.push_back(dim.fixed_sizes[dist(random_generator_)]);
            } else {
                // ，1
                shape.push_back(1);
            }
        }
    }
    
    return shape;
}

std::vector<std::vector<std::vector<int>>> AutoTuner::generate_sample_shapes() const {
    std::vector<std::vector<std::vector<int>>> samples;
    
    // 
    for (const auto& shape_config : config_.shape_configs) {
        std::vector<std::vector<int>> input_samples;
        
        switch (config_.shape_strategy) {
            case DynamicShapeStrategy::FIXED: {
                // 
                std::vector<int> fixed_shape;
                for (const auto& dim : shape_config.dimensions) {
                    fixed_shape.push_back(dim.fixed_sizes.empty() ? dim.min_size : dim.fixed_sizes[0]);
                }
                input_samples.push_back(fixed_shape);
                break;
            }
            
            case DynamicShapeStrategy::DYNAMIC: {
                // 
                const int num_samples = 10;
                for (int i = 0; i < num_samples; ++i) {
                    input_samples.push_back(generate_random_shape(shape_config));
                }
                break;
            }
            
            case DynamicShapeStrategy::OPTIMIZE_SIZES: {
                // 
                for (const auto& dim : shape_config.dimensions) {
                    if (!dim.fixed_sizes.empty()) {
                        for (int size : dim.fixed_sizes) {
                            std::vector<int> shape(shape_config.dimensions.size(), 1);
                            shape[0] = size;  // 
                            input_samples.push_back(shape);
                        }
                    }
                }
                break;
            }
            
            case DynamicShapeStrategy::PADDING_OPTIMIZE: {
                // 
                const int num_samples = 10;
                for (int i = 0; i < num_samples; ++i) {
                    auto shape = generate_random_shape(shape_config);
                    if (shape_config.enable_padding) {
                        shape = optimize_padding(shape);
                    }
                    input_samples.push_back(shape);
                }
                break;
            }
        }
        
        samples.push_back(input_samples);
    }
    
    return samples;
}

std::vector<int> AutoTuner::optimize_padding(const std::vector<int>& shape) const {
    std::vector<int> padded_shape = shape;
    
    // 
    for (const auto& config : config_.shape_configs) {
        if (shape.size() == config.dimensions.size() && config.enable_padding) {
            // 
            for (size_t i = 0; i < shape.size(); ++i) {
                const auto& dim = config.dimensions[i];
                if (dim.is_dynamic) {
                    // 
                    int size = shape[i];
                    int padding = (config.padding_multiple - (size % config.padding_multiple)) % config.padding_multiple;
                    padded_shape[i] = size + padding;
                }
            }
            break;
        }
    }
    
    return padded_shape;
}

std::vector<std::vector<std::vector<int>>> AutoTuner::optimize_batching(
    const std::vector<std::vector<int>>& inputs) const {
    
    std::vector<std::vector<std::vector<int>>> batched_inputs;
    
    if (!config_.optimize_batch_size || inputs.empty()) {
        return {inputs};
    }
    
    // 
    for (int batch_size : config_.target_batch_sizes) {
        std::vector<std::vector<int>> current_batch;
        
        // 
        for (const auto& input : inputs) {
            current_batch.push_back(input);
            
            if (current_batch.size() == static_cast<size_t>(batch_size)) {
                batched_inputs.push_back(current_batch);
                current_batch.clear();
            }
        }
        
        // 
        if (!current_batch.empty()) {
            // 
            while (current_batch.size() < static_cast<size_t>(batch_size)) {
                current_batch.push_back(current_batch.back());
            }
            batched_inputs.push_back(current_batch);
        }
    }
    
    return batched_inputs;
}

//  benchmark_model 
std::unordered_map<std::string, float> AutoTuner::benchmark_model(
    Model* model,
    const std::unordered_map<std::string, float>& params,
    const std::vector<std::vector<int>>& sample_inputs) {
    
    // Apply parameters to model
    if (!apply_best_parameters(model, params)) {
        throw std::runtime_error("Failed to apply parameters for benchmarking");
    }
    
    // 
    auto shape_samples = generate_sample_shapes();
    
    // ，
    std::vector<std::vector<std::vector<int>>> all_inputs;
    if (config_.optimize_batch_size) {
        for (const auto& input_samples : shape_samples) {
            auto batched = optimize_batching(input_samples);
            all_inputs.insert(all_inputs.end(), batched.begin(), batched.end());
        }
    } else {
        all_inputs = shape_samples;
    }
    
    // Warmup runs
    for (int i = 0; i < config_.warmup_runs; ++i) {
        for (const auto& input_set : all_inputs) {
            for (const auto& input : input_set) {
                model->run(input);
            }
        }
    }
    
    // Performance testing
    std::vector<float> latencies;
    float total_tokens = 0;
    float peak_memory = 0;
    float total_accuracy = 0;
    int total_samples = 0;
    
    for (int run = 0; run < config_.benchmark_runs; ++run) {
        for (const auto& input_set : all_inputs) {
            // Record start time
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Record initial memory usage
            float initial_memory = model->get_memory_usage();
            
            // Run inference
            std::vector<std::vector<int>> outputs;
            for (const auto& input : input_set) {
                outputs.push_back(model->run(input));
            }
            
            // Record end time
            auto end_time = std::chrono::high_resolution_clock::now();
            
            // Record peak memory
            float current_memory = model->get_memory_usage();
            peak_memory = std::max(peak_memory, current_memory - initial_memory);
            
            // Calculate latency
            float latency = std::chrono::duration<float, std::milli>(end_time - start_time).count();
            latencies.push_back(latency);
            
            // Accumulate token count and accuracy
            for (size_t i = 0; i < input_set.size(); ++i) {
                total_tokens += input_set[i].size();
                
                if (!outputs[i].empty() && model->has_ground_truth()) {
                    total_accuracy += model->calculate_accuracy(outputs[i]);
                    ++total_samples;
                }
            }
        }
    }
    
    // Calculate statistics
    float avg_latency = 0.0f;
    if (!latencies.empty()) {
        // Remove highest and lowest latencies
        if (latencies.size() > 2) {
            std::sort(latencies.begin(), latencies.end());
            float sum = 0.0f;
            for (size_t i = 1; i < latencies.size() - 1; ++i) {
                sum += latencies[i];
            }
            avg_latency = sum / (latencies.size() - 2);
        } else {
            avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
        }
    }
    
    // Calculate throughput (tokens/second)
    float total_time = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / 1000.0f;  // Convert to seconds
    float throughput = total_tokens / total_time;
    
    // Calculate average accuracy
    float accuracy = total_samples > 0 ? (total_accuracy / total_samples) * 100.0f : 0.0f;
    
    return {
        {"latency", avg_latency},
        {"throughput", throughput},
        {"memory", peak_memory},
        {"accuracy", accuracy}
    };
}

void AutoTuner::set_quantization_config(const QuantizationConfig& config) {
    // Add quantization config to tuning config
    config_.quantization = config;
}

QuantizationConfig AutoTuner::get_quantization_config() const {
    return config_.quantization;
}

QuantizationResult AutoTuner::quantize_model(
    Model* model,
    const std::vector<std::vector<int>>& calibration_data) {
    
    QuantizationResult result;
    
    try {
        if (!model) {
            throw std::runtime_error("Invalid model pointer");
        }

        // Record original FP32 accuracy
        result.accuracy_fp32 = evaluate_parameters(model, {}, calibration_data);
        
        // Calibrate quantization parameters
        if (!calibrate_quantization(model, calibration_data)) {
            throw std::runtime_error("Quantization calibration failed");
        }
        
        // Record initial memory and performance
        float initial_memory = model->get_memory_usage();
        auto initial_perf = benchmark_model(model, {}, calibration_data);
        
        // Apply quantization based on method
        switch (config_.quantization.method) {
            case QuantizationMethod::INT8:
                model->quantize_int8(config_.quantization.per_channel);
                break;
            case QuantizationMethod::FP16:
                model->quantize_fp16();
                break;
            case QuantizationMethod::INT4:
                model->quantize_int4(config_.quantization.per_channel);
                break;
            case QuantizationMethod::INT16:
                model->quantize_int16(config_.quantization.per_channel);
                break;
            case QuantizationMethod::DYNAMIC:
                model->quantize_dynamic(config_.quantization.num_bits);
                break;
            case QuantizationMethod::MIXED:
                model->quantize_mixed(config_.quantization.excluded_ops);
                break;
            default:
                throw std::runtime_error("Unsupported quantization method");
        }
        
        // Evaluate quantized model
        result = evaluate_quantization(model, calibration_data);
        
        // Calculate memory reduction
        float final_memory = model->get_memory_usage();
        result.memory_reduction = 1.0f - (final_memory / initial_memory);
        
        // Calculate speed improvement
        auto final_perf = benchmark_model(model, {}, calibration_data);
        result.speed_up = final_perf["throughput"] / initial_perf["throughput"];
        
        result.success = true;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    
    return result;
}

bool AutoTuner::calibrate_quantization(
    Model* model,
    const std::vector<std::vector<int>>& calibration_data) {
    
    try {
        // Select calibration subset based on ratio
        size_t calib_size = static_cast<size_t>(calibration_data.size() * config_.quantization.calib_ratio);
        std::vector<std::vector<int>> calib_subset(
            calibration_data.begin(),
            calibration_data.begin() + std::min(calib_size, calibration_data.size())
        );
        
        // Initialize calibration statistics
        std::unordered_map<std::string, std::vector<float>> stats;
        
        // Collect activation statistics
        for (const auto& input : calib_subset) {
            // Run inference and collect layer outputs
            auto layer_outputs = model->run_with_intermediate(input);
            
            // Update statistics based on calibration method
            for (const auto& [layer_name, output] : layer_outputs) {
                switch (config_.quantization.calib_method) {
                    case CalibrationMethod::MINMAX: {
                        auto [min_val, max_val] = model->get_tensor_range(output);
                        stats[layer_name] = {min_val, max_val};
                        break;
                    }
                    case CalibrationMethod::KL_DIVERGENCE:
                        stats[layer_name] = model->compute_kl_divergence(output);
                        break;
                    case CalibrationMethod::MSE:
                        stats[layer_name] = model->compute_mse_stats(output);
                        break;
                    case CalibrationMethod::ENTROPY:
                        stats[layer_name] = model->compute_entropy_stats(output);
                        break;
                    case CalibrationMethod::PERCENTILE:
                        stats[layer_name] = model->compute_percentile_stats(output);
                        break;
                }
            }
        }
        
        // Set calibration parameters
        for (const auto& [layer_name, layer_stats] : stats) {
            model->set_quantization_params(layer_name, layer_stats);
        }
        
        // Apply custom scales if provided
        for (const auto& scale_info : config_.quantization.custom_scales) {
            model->set_custom_scale(scale_info);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (config_.verbose) {
            std::cerr << "Calibration failed: " << e.what() << std::endl;
        }
        return false;
    }
}

QuantizationResult AutoTuner::evaluate_quantization(
    Model* model,
    const std::vector<std::vector<int>>& test_data) {
    
    QuantizationResult result;
    
    try {
        // Evaluate accuracy on test data
        float total_accuracy = 0.0f;
        std::unordered_map<std::string, float> layer_errors;
        
        for (const auto& input : test_data) {
            // Run inference and collect metrics
            auto output = model->run(input);
            auto layer_metrics = model->get_layer_metrics();
            
            // Update accuracy
            if (model->has_ground_truth()) {
                total_accuracy += model->calculate_accuracy(output);
            }
            
            // Accumulate layer-wise errors
            for (const auto& [layer_name, error] : layer_metrics) {
                layer_errors[layer_name] += error;
            }
        }
        
        // Calculate average accuracy
        result.accuracy_quantized = total_accuracy / test_data.size();
        
        // Calculate average layer errors
        for (auto& [layer_name, error] : layer_errors) {
            error /= test_data.size();
        }
        result.layer_wise_errors = layer_errors;
        
        // Check if accuracy degradation is within tolerance
        float accuracy_drop = result.accuracy_fp32 - result.accuracy_quantized;
        result.success = accuracy_drop <= config_.quantization.tolerance;
        
        if (!result.success) {
            result.error_message = "Accuracy degradation exceeds tolerance";
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    
    return result;
}

} // namespace deeppowers 