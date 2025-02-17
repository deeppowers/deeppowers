#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>

namespace deeppowers {

class ThreadPool {
public:
    // Constructor starts the thread pool
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    
    // Destructor joins all threads
    ~ThreadPool();

    // Add new work item to the pool
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    // Get the number of threads in the pool
    size_t size() const { return workers_.size(); }

    // Wait for all tasks to complete
    void wait_all();

private:
    // Need to keep track of threads so we can join them
    std::vector<std::thread> workers_;
    
    // The task queue
    std::queue<std::function<void()>> tasks_;
    
    // Synchronization
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
    
    // Track active tasks
    size_t active_tasks_;
    std::mutex active_mutex_;
    std::condition_variable active_condition_;
};

// Template implementation
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Don't allow enqueueing after stopping the pool
        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        {
            std::unique_lock<std::mutex> active_lock(active_mutex_);
            active_tasks_++;
        }
        
        tasks_.emplace([task](){ (*task)(); });
    }
    condition_.notify_one();
    return res;
}

} // namespace deeppowers 