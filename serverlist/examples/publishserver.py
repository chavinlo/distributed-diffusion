import json
import requests

url = "http://localhost:5000/servers"

"""
dataset types:
coordinated - connects to a server to determine what chunks to train on
random - chosses randomly from a given list (such as sqlite db)

nsfw:
0 - No NSFW content
1 - NSFW content
2 - Maybe NSFW content
"""

data = {
    "info": {
        "name": "Awesome Finetune",
        "author": "johndoe95",
        "desc": "Let's join together to finetune SD!",
        "nsfw": 2
    },
    "peers": 0,
    "resources": {
        "CPUs": 0,
        "GPUs": 0,
        "TFLOPs": 0,
    },
    "trainer_script": "stablediffusion/text2image.py",
    "swarm_config": {
        "dht": {
            "type": "single",
            "entrypoint": "/ip4/127.0.0.1/tcp/XXX/p2p/YYY"
        },
        "data": {
            "type": "random",
            "source": "sqlite-db",
            "path": "http://127.0.0.1/dataset.db"
        }
    },
    "trainer_config": {
        "train_text_encoder": True,
        "learning_rate": 0.001
    },
    "status": "waiting"
}

headers = {'Content-type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.status_code)
print(response.json())
