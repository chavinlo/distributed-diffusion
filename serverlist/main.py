from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import json

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///server_list.db"
db = SQLAlchemy(app)

class Swarm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    info = db.Column(db.String)
    peers = db.Column(db.Integer)
    resources = db.Column(db.String)
    trainer_script = db.Column(db.String)
    swarm_config = db.Column(db.String)
    trainer_config = db.Column(db.String)
    status = db.Column(db.String)

with app.app_context():
    db.create_all()

@app.route("/servers", methods=["GET"])
def get_servers():
    servers = Swarm.query.all()
    server_list = []
    for server in servers:
        server_list.append({
            "id": server.id, 
            "info": json.loads(server.info),
            "peers": server.peers, 
            "resources": json.loads(server.resources), 
            "trainer_script": server.trainer_script, 
            "status": server.status})
    return jsonify(server_list)

@app.route("/servers/<int:id>", methods=["GET"])
def get_server(id):
    server = Swarm.query.get(id)
    return jsonify({
        "id": server.id, 
        "info": json.loads(server.info),
        "peers": server.peers, 
        "resources": json.loads(server.resources), 
        "trainer_script": server.trainer_script, 
        "swarm_config": json.loads(server.swarm_config), 
        "trainer_config": json.loads(server.trainer_config), 
        "status": server.status})

@app.route("/servers", methods=["POST"])
def add_server():
    data = request.get_json()
    new_server = Swarm(
        info=json.dumps(data['info']),
        peers=data["peers"], 
        resources=json.dumps(data["resources"]), 
        trainer_script=data["trainer_script"], 
        swarm_config=json.dumps(data["swarm_config"]), 
        trainer_config=json.dumps(data["trainer_config"]), 
        status=data["status"])
    db.session.add(new_server)
    db.session.commit()
    return jsonify({"message": "Server added successfully"})

@app.route("/servers/<int:id>", methods=["PUT"])
def update_server(id):
    server = Swarm.query.get(id)
    data = request.get_json()
    server.info = json.dumps(data['info'])
    server.peers = data["peers"]
    server.resources = json.dumps(data["resources"])
    server.trainer_script = data["trainer_script"]
    server.swarm_config = json.dumps(data["swarm_config"])
    server.trainer_config = json.dumps(data["trainer_config"])
    server.status = data["status"]
    db.session.commit()
    return jsonify({"message": "Server updated successfully"})

@app.route("/servers/<int:id>", methods=["DELETE"])
def delete_server(id):
    server = Swarm.query.get(id)
    db.session.delete(server)
    db.session.commit()
    return jsonify({"message": "Server deleted successfully"})

if __name__ == "__main__":
    app.run(debug=True)

