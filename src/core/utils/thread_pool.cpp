#include "thread_pool.hpp"

namespace deeppowers {

ThreadPool::ThreadPool(size_t num_threads)
    : stop_(false)
    , active_tasks_(0) {
    
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] {
                        return stop_ || !tasks_.empty();
                    });
                    
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                task();
                
                {
                    std::unique_lock<std::mutex> lock(active_mutex_);
                    if (--active_tasks_ == 0) {
                        active_condition_.notify_all();
                    }
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_condition_.wait(lock, [this] {
        return active_tasks_ == 0;
    });
}

} // namespace deeppowers 