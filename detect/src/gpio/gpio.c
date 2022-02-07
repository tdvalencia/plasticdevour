#include <gpio.h>

void test() {
	DEV_ModuleInit();
	Motor_Init();
	Motor_Run(MOTORA, BACKWARD, 75);

}

int init_gpio() {
	if (DEV_ModuleInit())
		return 1;		
	Motor_Init();
	return 0;
}

void gpio_on() {
	Motor_Run(MOTORA, BACKWARD, 75);
}

void gpio_off() {
	Motor_Stop(MOTORA);
}

void cleanup() {
	Motor_Stop(MOTORA);
	DEV_ModuleExit();
}
