from pyfirmata import Arduino
import time
from firebase import firebase
board = Arduino("COM3")
firebase = firebase.FirebaseApplication("https://python-7521f.firebaseio.com/python-7521f", None)
def led_on():
    board.digital[13].write(1)
    print("Led is on")

def led_off():
    board.digital[13].write(0)
    print("led id of")

while 1:
    result = firebase.get('/user/id/command', None)
    print(result)
    if result=="lights on":
        led_on()
    elif result =="lights off":
        led_off()
