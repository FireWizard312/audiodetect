from wsgiref.simple_server import make_server
import falcon
import light_tasks
        
# Falcon follows the REST architectural style, meaning (among
# other things) that you think in terms of resources and state
# transitions, which map to HTTP verbs.
class LightsUpdate:
    def on_get(self, req, resp):
        class_id = req.get_param('class_id')
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        class_id = int(class_id)
        if class_id == 8:
            light_tasks.sirenlightson.delay()
        elif class_id == 1:
            light_tasks.honklightson.delay()
        else:
            light_tasks.lightsoff.delay()
                    

# falcon.App instances are callable WSGI apps
# in larger applications the app is created in a separate file
app = falcon.App()

# Resources are represented by long-lived class instances
lights = LightsUpdate()

# things will handle all requests to the '/things' URL path
app.add_route('/', lights)

if __name__ == '__main__':
    with make_server('', 8000, app) as httpd:
        print('Serving on port 8000...')

        # Serve until process is killed
        httpd.serve_forever()
