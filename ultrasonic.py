import RPi.GPIO as GPIO
import time
import statistics

#### Define program constants
trigger_pin=4    # the GPIO pin that is set to high to send an ultrasonic wave out. (output)
echo_pin=17      # the GPIO pin that indicates a returning ultrasonic wave when it is set to high (input)
number_of_samples=5 # this is the number of times the sensor tests the distance and then picks the middle value to return
sample_sleep = .01  # amount of time in seconds that the system sleeps before sending another sample request to the sensor. You can try this at .05 if your measurements aren't good, or try it at 005 if you want faster sampling.
calibration1 = 30   # the distance the sensor was calibrated at
calibration2 = 1750 # the median value reported back from the sensor at 30 cm
time_out = .05 # measured in seconds in case the program gets stuck in a loop

#### Set up GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Set up the pins for output and input
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#### initialize variables
samples_list = [] #type: list # list of data collected from sensor which are averaged for each measurement
stack = []


def timer_call(channel) :
# call back function when the rising edge is detected on the echo pin
    now = time.monotonic()  # gets the current time with a lot of decimal places
    stack.append(now) # stores the start and end times for the distance measurement in a LIFO stack

def trigger():
    # set our trigger high, triggering a pulse to be sent - a 1/100,000 of a second pulse or 10 microseconds
    GPIO.output(trigger_pin, GPIO.HIGH) 
    time.sleep(0.00001) 
    GPIO.output(trigger_pin, GPIO.LOW)

def check_distance():
# generates an ultrasonic pulse and uses the times that are recorded on the stack to calculate the distance
    # Empty the samples list
    samples_list.clear()

    while len(samples_list) < number_of_samples:       # Checks if the samples_list contains the required number_of_samples
        # Tell the sensor to send out an ultrasonic pulse.
        trigger()

        # check the length of stack to see if it contains a start and end time . Wait until 2 items in the list
        while len(stack) < 2:                          # waiting for the stack to fill with a start and end time
            start = time.monotonic()                   # get the time that we enter this loop to track for timeout
            while time.monotonic() < start + time_out: # check the timeout condition
                pass

            trigger()                                  # The system timed out waiting for the echo to come back. Send a new pulse.

        if len(stack) == 2:                          # Stack has two elements on it.
            # once the stack has two elements in it, store the difference in the samples_list
            samples_list.append(stack.pop()-stack.pop())

        elif len(stack) > 2:
            # somehow we got three items on the stack, so clear the stack
            stack.clear()

        time.sleep(sample_sleep)          # Pause to make sure we don't overload the sensor with requests and allow the noise to die down

    # returns the media distance calculation
    
    # Capture the current time
    measurement_time = time.monotonic()

    # Calculate and return distance and the current time
    distance = (statistics.median(samples_list)*1000000*calibration1/calibration2)
    return distance, measurement_time

###########################
# Main Program
###########################

GPIO.add_event_detect(echo_pin, GPIO.BOTH, callback=timer_call)  # add rising and falling edge detection on echo_pin (input)

previous_distance, previous_time = check_distance()
previous_speed = 0

for i in range(1000):
    distance, current_time = check_distance()
    time_interval = current_time - previous_time
    speed = (distance - previous_distance) / time_interval
    acceleration = (speed - previous_speed) / time_interval

    print(f"Distance: {round(distance, 1)} cm, Speed: {speed:.2f} cm/s, Acceleration: {acceleration:.2f} cm/s²")

    previous_distance = distance
    previous_time = current_time
    previous_speed = speed
    time.sleep(sample_sleep)
