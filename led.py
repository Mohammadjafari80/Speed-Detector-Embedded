import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin to which the LED is connected (via a resistor)
LED_PIN = 18

# Set up the GPIO pin as an output
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    while True:
        # Turn off the LED by setting the GPIO pin to HIGH
        # Because it's connected to 3.3V, setting it to HIGH will turn it off
        GPIO.output(LED_PIN, GPIO.HIGH)
        print("LED OFF")
        time.sleep(1)  # LED will stay off for 1 second

        # Turn on the LED by setting the GPIO pin to LOW
        # This will create a circuit through ground and light up the LED
        GPIO.output(LED_PIN, GPIO.LOW)
        print("LED ON")
        time.sleep(1)  # LED will stay on for 1 second

except KeyboardInterrupt:
    # Clean up the GPIO on CTRL+C exit
    GPIO.cleanup()

# Clean up the GPIO on normal exit
GPIO.cleanup()
