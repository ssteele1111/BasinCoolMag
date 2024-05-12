/*
 * timer.h
 *
 *  Created on: Sep 12, 2020
 *      Author: eai
 */


#include <iostream>
#include <chrono>  // for high_resolution_clock


#ifndef TIMER_H_
#define TIMER_H_

class Timer
{
public:
  Timer(const std::string& name)
    : name_ (name),
      start_ (std::clock())
    {
    }
  ~Timer()
    {
      double elapsed = (double(std::clock() - start_) / double(CLOCKS_PER_SEC));
      std::cout << "# " << name_ << ": " << int(elapsed * 1000) << " ms" << std::endl;
    }
private:
  std::string name_;
  std::clock_t start_;
};

#endif /* TIMER_H_ */

