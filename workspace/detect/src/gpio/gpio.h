#pragma once

#include <wiringPi.h>

void init_gpio();
void gpio_on();
void gpio_off();
void cleanup();
