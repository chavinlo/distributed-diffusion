import requests

def get_config():
    r = requests.get("http://127.0.0.1:5000/internal/config")
    r = r.json()
    