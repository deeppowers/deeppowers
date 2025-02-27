#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "model_types.hpp"
#include "tensor.hpp"

namespace deeppowers {

/**
 * Base model class for inference
 */
class Model {
public:
    /**
     * Default constructor
     */
    Model();
    
    /**
     * Virtual destructor
     */
    virtual ~Model();
    
    /**
     * Run inference on input tensor
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor forward(const Tensor& input);
    
    /**
     * Run inference on batch of input tensors
     * @param inputs Vector of input tensors
     * @return Vector of output tensors
     */
    virtual std::vector<Tensor> forward_batch(const std::vector<Tensor>& inputs);
    
    /**
     * Save model to file
     * @param path Output file path
     * @param format Output format (default: same as input)
     */
    virtual void save(const std::string& path, 
                     ModelFormat format = ModelFormat::AUTO);
    
    /**
     * Get model configuration
     * @return Model configuration map
     */
    virtual std::unordered_map<std::string, std::string> config() const;
    
    /**
     * Get model device
     * @return Device name
     */
    virtual std::string device() const;
    
    /**
     * Move model to device
     * @param device_name Target device name
     */
    virtual void to(const std::string& device_name);
    
    /**
     * Get model type
     * @return Model type
     */
    virtual std::string model_type() const;
    
    /**
     * Get model precision
     * @return Model precision
     */
    virtual PrecisionMode precision() const;
    
    /**
     * Set model precision
     * @param precision Target precision
     */
    virtual void set_precision(PrecisionMode precision);
    
    /**
     * Get model optimization level
     * @return Optimization level
     */
    virtual OptimizationLevel optimization_level() const;
    
    /**
     * Set model optimization level
     * @param level Target optimization level
     */
    virtual void set_optimization_level(OptimizationLevel level);
};

} // namespace deeppowers 