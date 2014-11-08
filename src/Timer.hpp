#pragma once

#include <iostream>
#include <chrono>

/**
 * Extremely simple Timer class using C++11 chrono library
**/
class Timer
{
	public:
	/**
	 * @param run Pass true to start the Timer right away
	**/
	explicit Timer(bool run = false) : 
		_running(run)
	{
		if(_running)
			start();
	}
	
	inline void start()
	{
		reset();
		_running = true;
	}
	
	inline void reset()
	{
		_start = std::chrono::high_resolution_clock::now();
		_pause = _start;
	}
	
	inline void pause()
	{
		_pause = currentTimePoint();
		_running = false;
	}
	
	inline void unpause()
	{
		if(_running)
			return;
		
		_start += currentTimePoint() - _pause;
		_running = true;
	}
	
	/**
	 * @return Running time in Unit
	**/
	template<typename Unit>
	inline Unit runningTime() const
	{
		return std::chrono::duration_cast<Unit>(runningTime());
	}
	
	/**
	 * @return Running time in nanoseconds
	**/
	inline std::chrono::nanoseconds nanoseconds() const
	{
		return runningTime<std::chrono::nanoseconds>();
	}
	
	/**
	 * @return Running time in microseconds
	**/
	inline std::chrono::microseconds microseconds() const
	{
		return runningTime<std::chrono::microseconds>();
	}
	
	/**
	 * @return Running time in milliseconds
	**/
	inline std::chrono::milliseconds milliseconds() const
	{
		return runningTime<std::chrono::milliseconds>();
	}
	
	/**
	 * @return Running time in seconds
	**/
	inline std::chrono::seconds seconds() const
	{
		return runningTime<std::chrono::seconds>();
	}
	
	template <typename CharT, typename Traits>
	inline friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out, const Timer& t)
	{
		return out << t.milliseconds().count();
	}
	
	private:
	bool											_running;
	std::chrono::high_resolution_clock::time_point	_start;
	std::chrono::high_resolution_clock::time_point	_pause;
	
	inline std::chrono::high_resolution_clock::time_point currentTimePoint() const 
	{ return std::chrono::high_resolution_clock::now(); }
	
	inline std::chrono::high_resolution_clock::duration runningTime() const
	{
		return (_running) ?
			currentTimePoint() - _start :
			_pause - _start;
	}
};