from celery import Celery
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
import time
from wsgiref.simple_server import make_server
import falcon

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

inspect = app.control.inspect()
def removetask():
        taskid = inspect.active()
        taskid = taskid[0]
        app.control.revoke(taskid, terminate = True)
        
# Falcon follows the REST architectural style, meaning (among
# other things) that you think in terms of resources and state
# transitions, which map to HTTP verbs.
class ThingsResource:
    def on_get(self, req, resp):
        class_id = req.get_param('class_id')
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        class_id = int(class_id)
        if class_id == 8:
            lightson.delay()
        else:
            removetask()
            lightsoff.delay()
                    

# falcon.App instances are callable WSGI apps
# in larger applications the app is created in a separate file
app = falcon.App()

# Resources are represented by long-lived class instances
things = ThingsResource()

# things will handle all requests to the '/things' URL path
app.add_route('/', things)

if __name__ == '__main__':
    with make_server('', 8000, app) as httpd:
        print('Serving on port 8000...')

        # Serve until process is killed
        httpd.serve_forever()
