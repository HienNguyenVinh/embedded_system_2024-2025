# door_control.py
import RPi.GPIO as GPIO
import time

class DoorController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        
        self.servo_pwm = GPIO.PWM(SERVO_PIN, 50)
        self.servo_pwm.start(0)

    def set_servo_angle(angle: float):
        duty = (angle / 18.0) + 2.0
        servo_pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)       
        servo_pwm.ChangeDutyCycle(0) 

    def open_door(self):
        try:
            self.set_servo_angle(90)
            time.sleep(5)
            self.set_servo_angle(0)
        except Exception as e:
            print("Error in open_door:", e)

    def is_ultrasonic_triggered(self, threshold_cm=30.0):
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.05)
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        pulse_start = time.time()
        timeout = pulse_start + 0.04
        while GPIO.input(ECHO_PIN) == 0 and time.time() < timeout:
            pulse_start = time.time()

        pulse_end = time.time()
        timeout = pulse_end + 0.04
        while GPIO.input(ECHO_PIN) == 1 and time.time() < timeout:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance_cm = (pulse_duration * 34300) / 2

        # print(f"Distance: {distance_cm:.1f} cm")
        return distance_cm <= threshold_cm

    def activate_buzzer(self, beeps=3, on_time=0.5, off_time=0.5):
        for _ in range(beeps):
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(on_time)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(off_time)

    def cleanup(self):
        self.servo_pwm.stop()
        GPIO.cleanup()
