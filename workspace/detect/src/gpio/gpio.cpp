#include <wiringPi.h>
#include <gpio.h>

void init_gpio() {
	wiringPiSetupGpio();
	pinMode(18, OUTPUT);
}

void gpio_on() {
	digitalWrite(18, HIGH);
}

void gpio_off() {
	digitalWrite(18, LOW);
}

void cleanuo() {

}
