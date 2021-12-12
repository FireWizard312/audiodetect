from celery import Celery
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
import time

app = Celery()

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial)

siren = [(0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(2,6),(3,6),(4,6),(5,6),(2,5),(3,5),(4,5),(5,5),(2,4),(3,4),(4,4),(5,4),(3,3),(4,3),(1,0),(2,1),(5,1),(6,0)]
honk = [(0,3),(1,3),(0,4),(1,4),(0,2),(1,2),(0,5),(1,5),(2,3),(3,3),(4,3),(2,4),(3,4),(4,4),(4,2),(4,5),(6,2),(7,1),(6,5),(7,6)]

@app.task
def lightson():
    while True:
        with canvas(device) as draw:
            draw.point(siren, fill = "white")
        time.sleep(0.75)
        with canvas(device) as draw:
            draw.point(siren, fill = "black")
        time.sleep(0.5)

@app.task
def lightsoff():
    with canvas(device) as draw:
        draw.point(siren, fill = "black")

def removetask():
        taskid = app.control.inspect.active()
        taskid = taskid[0]
        app.control.revoke(taskid, terminate = True)