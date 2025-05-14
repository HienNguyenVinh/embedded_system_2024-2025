# door_control.py
import RPi.GPIO as GPIO
from gpiozero import Buzzer, DistanceSensor, AngularServo
import time


class DoorController:
    def __init__(self):
        SERVO_PIN   = 15
        TRIG_PIN    = 8
        ECHO_PIN    = 9 
        BUZZER_PIN  = 17
        
        self.distance_sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN)
        # self.servo = AngularServo(SERVO_PIN, min_angle=0, max_angle=90, initial_angle=0)
        self.buzzer = Buzzer(BUZZER_PIN)

    def open_door(self):
        try:
            SERVO_PIN   = 15
            self.servo = AngularServo(SERVO_PIN, min_angle=0, max_angle=90)
            self.servo.angle = 90
            print("Opening door...")
            time.sleep(3)
            self.servo.angle = 0 
            print("Door opened and closed successfully.")
            
        except Exception as e:
            print("Error in open_door:", e)

    def is_ultrasonic_triggered(self, threshold_cm=30.0):
        return False

    def activate_buzzer(self, beeps=3, on_time=0.5, off_time=0.5):
        for _ in range(beeps):
            self.buzzer.on()
            time.sleep(on_time)
            self.buzzer.off()
            time.sleep(off_time)

# if __name__ == "__main__":
#     try:
#         door_controller = DoorController()
        
#         door_controller.activate_buzzer()
#         door_controller.open_door()
#     except KeyboardInterrupt:
#         print("Program interrupted.")
#         pass