import requests

url = "http://192.168.0.141:8000/things"
req = requests.get(url)
print(req.content)