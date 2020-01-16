#pragma once
#include <chrono>

class timer
{
public:
	/**
	 * \brief Starts a new timer
	 */
	timer() :start_time_(std::chrono::steady_clock::now()) {}

	/**
	 * \brief Stops the timer (if still running) and returns the recorduration
	 * \return Duration in ms that elapsed between construction and first call of get()
	 */
	long long get()
	{
		if (!ended_) {
			end_time_ = std::chrono::steady_clock::now();
			ended_ = true;
		}
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_);
		return duration.count();
	}

private:
	std::chrono::steady_clock::time_point start_time_;
	std::chrono::steady_clock::time_point end_time_;
	bool ended_ = false;
};