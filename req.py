import requests
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
import time
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial)

for i in range(3):
    with canvas(device) as draw:
        draw.point((3,3), fill = "white")
        draw.point((3,4), fill = "white")
        draw.point((4,4), fill = "white")
        draw.point((4,3), fill = "white")
    time.sleep(2)
    with canvas(device) as draw:
        draw.point((3,3), fill = "black")
        draw.point((3,4), fill = "black")
        draw.point((4,4), fill = "black")
        draw.point((4,3), fill = "black")
    time.sleep(2)
# url = "http://192.168.0.141:8000/things"
# req = requests.get(url)
# print(req.content)
