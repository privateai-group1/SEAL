#pragma once
#include <string>
#include <chrono>
#include <iostream>

class stopwatch
{
public:
	stopwatch(std::string timer_name) :
		name_(timer_name),
		start_time_(std::chrono::steady_clock::now())
	{
	}

	~stopwatch()
	{
		auto end_time = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
		std::cout << name_ << ": " << duration.count() << " milliseconds" << std::endl;
	}

private:
	std::string name_;
	std::chrono::steady_clock::time_point start_time_;
};
