#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

#include "DEV_Config.h"
#include <time.h>
#include "DEV_Config.h"
#include "MotorDriver.h"

void test();
int init_gpio();
void gpio_on();
void gpio_off();
void cleanup();
