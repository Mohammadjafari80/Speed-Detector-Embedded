import RPi.GPIO as GPIO

# Use BCM GPIO references instead of physical pin numbers
GPIO.setmode(GPIO.BCM)

# Define your list of GPIO pins here
gpio_pins = [2, 3, 4, 5]

# Set up the GPIO pins as inputs with pull-down resistors (since we are reading their status)
for pin in gpio_pins:
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Read and print the status of each GPIO pin
for pin in gpio_pins:
    status = GPIO.input(pin)
    print(f"GPIO {pin}: {'HIGH' if status else 'LOW'}")

# Clean up GPIO settings
GPIO.cleanup()
