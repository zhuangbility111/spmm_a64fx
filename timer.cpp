#include "timer.h"

void Timer::start() {
    startTime = std::chrono::high_resolution_clock::now();
}

void Timer::end() {
    endTime = std::chrono::high_resolution_clock::now();
}

void Timer::print() {
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Elapsed time: " << elapsedTime << " milliseconds" << std::endl;
}
