import requests
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
import time
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial)

siren = [(0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(2,6),(3,6),(4,6),(5,6),(2,5),(3,5),(4,5),(5,5),(2,4),(3,4),(4,4),(5,4),(3,3),(4,3),(1,0),(2,1),(5,1),(6,0)]
honk = [(1,3),(2,3),(1,4),(2,4),(3,3),(4,3),(5,3),(3,4),(4,4),(5,5),(5,2),(5,5)]
for i in range(3):
    with canvas(device) as draw:
        draw.point(honk, fill = "white")
        # draw.point((2,6), fill = "white")
        # draw.point((3,6), fill = "white")
        # draw.point((4,6), fill = "white")
        # draw.point((5,6), fill = "white")
        # draw.point((6,6), fill = "white")
        # draw.point((2,5), fill = "white")
        # draw.point((3,5), fill = "white")
        # draw.point((4,5), fill = "white")
        # draw.point((5,5), fill = "white")
        # draw.point((2,4), fill = "white")
        # draw.point((3,4), fill = "white")
        # draw.point((4,4), fill = "white")
        # draw.point((5,4), fill = "white")
        # draw.point((2,3), fill = "white")
        # draw.point((3,3), fill = "white")
        # draw.point((4,3), fill = "white")
        # draw.point((5,3), fill = "white")
        # draw.point((3,2), fill = "white")
        # draw.point((4,2), fill = "white")
        # draw.point((1,1), fill = "white")
        # draw.point((0,0), fill = "white")
        # draw.point((6,1), fill = "white")
        # draw.point((7,0), fill = "white")
        # draw.point((0,4), fill = "white")
        # draw.point((7,4), fill = "white")

    time.sleep(2)
    with canvas(device) as draw:
        draw.point(honk, fill = "black")
    #     draw.point((2,6), fill = "black")
    #     draw.point((3,6), fill = "black")
    #     draw.point((4,6), fill = "black")
    #     draw.point((5,6), fill = "black")
    #     draw.point((6,6), fill = "black")
    #     draw.point((2,5), fill = "black")
    #     draw.point((3,5), fill = "black")
    #     draw.point((4,5), fill = "black")
    #     draw.point((5,5), fill = "black")
    #     draw.point((2,4), fill = "black")
    #     draw.point((3,4), fill = "black")
    #     draw.point((4,4), fill = "black")
    #     draw.point((5,4), fill = "black")
    #     draw.point((2,3), fill = "black")
    #     draw.point((3,3), fill = "black")
    #     draw.point((4,3), fill = "black")
    #     draw.point((5,3), fill = "black")
    #     draw.point((3,2), fill = "black")
    #     draw.point((4,2), fill = "black")
    #     draw.point((1,1), fill = "black")
    #     draw.point((0,0), fill = "black")
    #     draw.point((6,1), fill = "black")
    #     draw.point((7,0), fill = "black")
    #     draw.point((0,4), fill = "black")
    #     draw.point((7,4), fill = "black")
    time.sleep(2)
# url = "http://192.168.0.141:8000/things"
# req = requests.get(url)
# print(req.content)

