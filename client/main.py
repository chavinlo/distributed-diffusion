from flask import Flask, render_template, request, jsonify
import requests
import os

local_config = {}
config_to_send = {}

def get_all_items(path):
    items = []
    for root, dirs, files in os.walk(path):
        for name in files:
            items.append(os.path.join(root, name).replace("../trainers/", ""))
        for name in dirs:
            items.append(os.path.join(root, name).replace("../trainers/", ""))
    return items

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def root():
    return render_template('serverlist.html')

@app.route("/config/save", methods=["POST"])
def save_config():
    global local_config
    local_config = request.json
    return "Configuration saved."

@app.route('/join', methods=["POST"])
def join():
    server_id = request.args.get('server')
    response = requests.get(f'http://127.0.0.1:4000/servers/{server_id}')
    response = response.json()

    available_trainers = get_all_items('../trainers/')
    if response['trainer_script'] not in available_trainers:
        return jsonify({
            "info": "trainer not found",
            "available trainers": available_trainers
        })

    global config_to_send
    config_to_send = {
        "swarm_config": response['swarm_config'],
        "trainer_config": {
            "global": response['trainer_config'],
        }
    }
    
    return "ok"

@app.route("/internal/config")
def trainer_config():
    try:
        global config_to_send
        global local_config
        config_to_send['trainer_config']['local'] = local_config
        return jsonify(config_to_send)
    except KeyError:
        return "error config not set yet"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
