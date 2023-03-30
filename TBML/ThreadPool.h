
#pragma once
#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>


// https://github.com/progschj/ThreadPool
class ThreadPool
{
private:
	size_t threads;
	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;

	std::mutex queueMutex;
	std::condition_variable condition;
	bool stop;


public:
	ThreadPool(size_t threads = std::thread::hardware_concurrency()) : threads(threads), stop(false)
	{
		// Setup worker threads
		for (size_t i = 0; i < threads; ++i)
		{
			workers.emplace_back([this]
				{

					// Run indefinetly
					for (;;)
					{
						std::function<void()> task;

						// Lock the queue, and take top task
						{
							std::unique_lock<std::mutex> lock(this->queueMutex);
							this->condition.wait(lock,
								[this] { return this->stop || !this->tasks.empty(); });
							if (this->stop && this->tasks.empty()) return;
							task = std::move(this->tasks.front());
							this->tasks.pop();
						}

						// Run task
						task();
					}
				});
		}
	}


	~ThreadPool()
	{
		// Wait for queue lock
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			stop = true;
		}

		// Wait for all worker threads to finish
		condition.notify_all();
		for (std::thread& worker : workers) worker.join();
	}


	template<class F, class... Args>
	auto enqueue(F&& f, Args&&... args)
		-> std::future<typename std::result_of<F(Args...)>::type>
	{
		using return_type = typename std::result_of<F(Args...)>::type;

		// Setup task with function
		auto task = std::make_shared<std::packaged_task<return_type()>>(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
		);

		// Setup return future and lock queue
		std::future<return_type> res = task->get_future();
		{
			std::unique_lock<std::mutex> lock(queueMutex);

			// Add task to queue if not stopped
			if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
			tasks.emplace([task]() { (*task)(); });
		}

		// Notify and return result future
		condition.notify_one();
		return res;
	}


	size_t size() const { return threads; }
};