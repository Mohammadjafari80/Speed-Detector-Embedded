import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin to which the LED is connected (via a resistor)
LED_PIN_RED = 18
LED_PIN_GREEN = 22

# Set up the GPIO pin as an output
GPIO.setup(LED_PIN_RED, GPIO.OUT)
GPIO.setup(LED_PIN_GREEN, GPIO.OUT)


def turn_led_on(gp: int):
    GPIO.output(gp, GPIO.HIGH)
    print("LED OFF")

def turn_led_off(gp: int):
    GPIO.output(gp, GPIO.LOW)
    print("LED ON")