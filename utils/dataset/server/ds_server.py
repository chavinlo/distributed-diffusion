import sys
from flask import Flask, jsonify, request, send_file, Response
from pathlib import Path
from zipfile import ZipFile
import os
import argparse
import time
from io import BytesIO
from datetime import datetime
import threading


import sqlite3
import random

# Connect to the database
conn = sqlite3.connect('danbooru.db')
cursor = conn.cursor()

# Load the posts table into memory
cursor.execute('SELECT * FROM posts')
posts = cursor.fetchall()

def select_random_post():
  # Select a random record from the posts table
  random_post = random.choice(posts)
  post_id = random_post[0]
  image_ext = random_post[1]
  rating = random_post[2]
  
  # Return the post_id, image_ext, and rating
  return post_id, image_ext, rating

# Get the total number of rows in the posts table
cursor.execute('SELECT COUNT(*) FROM posts')
num_rows = cursor.fetchone()[0]

def get_num_rows():
    return num_rows

# Close the connection to the database
conn.close()


parser = argparse.ArgumentParser(description='Dataset server')
parser.add_argument('--dataset', type=str, default=None, required=True, help='Path to dataset')
#TODO: make these work
#parser.add_argument('--new', type=bool, default=True, help='re-scan the dataset folder')
#parser.add_argument('--load', type=str, default=None, help='path to the JSON DB snapshot')
parser.add_argument('--name', type=str, default="Dataset Server", help='Server name')
parser.add_argument('--description', type=str, default="Just a dataset server", required=False, help='Server description')
parser.add_argument('--tasktimeout', type=int, default=20, required=False, help='Time to wait for a task to be completed (in minutes)')
args = parser.parse_args()

#info
version = "v1"
execDate = str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
print("Server Version: " + version)
print("Current Time: " + execDate)

#gt/GetTime in seconds
def gt():
    return(time.time())

import hivemind

dht = hivemind.DHT(
host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
start=True,
daemon=True)

peerList = []

for addr in dht.get_visible_maddrs():
    peerList.append(str(addr))

print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

#this should not be used in the final version + might have directory traversal
def solvePath(filename):
    path = args.dataset + "/" + filename
    return(path)

def dictCreator(input):
    dataPath = Path(input)
    dataDict = os.listdir(dataPath)
    #sort everything
    sortedDict = {}
    entryId = 0
    for entry in dataDict:
        entryExt = os.path.splitext(entry)[1]
        entryFilename = os.path.splitext(entry)[0]
        #ignore txt files for now
        if entryExt == ".txt":
            continue
        if entryExt not in ('.jpg', '.webp', '.png', '.jpeg'):
            print("Not valid image format")
            continue
        expectedTxtName = str(entryFilename) + ".txt"
        #TODO: Change input to a proper Path object
        expectedTxtLocation = os.path.join(input + "/" + expectedTxtName)
        txtPairExists = os.path.isfile(expectedTxtLocation)
        #only add to dict the entries that have valid pairs (txt & img)
        if txtPairExists:
            tmpDict = {
                'imagefile': entry,
                'textfile': expectedTxtName,
                'assigned': False,
                'assignedExpirationDate': 'none',
                'epochs': 0,
                'entryId': entryId
            }
            sortedDict[entryId] = tmpDict
            entryId = entryId + 1
    print("Registered " + str(entryId) + " entries.")
    return sortedDict, entryId

#directory to the dataset
dataDir = args.dataset
filesDict, numberFiles = dictCreator(dataDir)


app = Flask(__name__)

#current version and info
@app.route("/")
def mainsite():
    info = {
        "ServerName": args.name,
        "ServerDescription": args.description,
        "ServerVersion": version,
        "FilesBeingServed": numberFiles,
        "ExecutedAt": execDate
    }
    return jsonify(info)

@app.route("/info")
def getInfo():
    info = {
        "ServerName": args.name,
        "ServerDescription": args.description,
        "ServerVersion": version,
        "FilesBeingServed": numberFiles,
        "ExecutedAt": execDate
    }
    return jsonify(info)

@app.route("/v1/get/peers", methods=['GET'])
def getpeers():
    print("Peer retrival")
    return jsonify(peerList)

#getTasksFull
@app.route("/v1/get/tasks/full")
def getTasksFull():
    return jsonify(filesDict)

#getTasks: return entries(objects) in dataset that need training
#it should return a list, that contains entries with low train count.
@app.route("/v1/get/tasks/<string:wantedTasks>")
#reverse=True to get descending
def getTasks(wantedTasks):
    #minus one cuz computer number system != human
    setMinutes = args.tasktimeout
    actualTime = gt()
    timeToExpire = actualTime + (60*setMinutes)
    intWantedTasks = int(wantedTasks) - 1
    if intWantedTasks > 3000:
        return(Response(status=404))
    listToReturn = []
    sortedDict = sorted(filesDict.items(), key=lambda x_y: x_y[1]['epochs'])
    obtainedTasks = 0
    x = 0
    while obtainedTasks < intWantedTasks:
        for i in sortedDict:
            if obtainedTasks > intWantedTasks:
                break
            if sortedDict[x][1]['assigned']:
                x = x + 1
                break
            listToReturn.append(sortedDict[x])
            entryId = sortedDict[x][0]
            filesDict[entryId]['assigned'] = True
            filesDict[entryId]['assignedExpirationDate'] = timeToExpire
            obtainedTasks = obtainedTasks + 1
            x = x + 1
    return jsonify(listToReturn)
    
@app.route('/v1/get/files', methods=['POST'])
def getFiles():
    print("Got request for files!")
    content = request.get_json(force=True)
    if len(content) > 3000:
        return(Response(status=404))
    memory_file = BytesIO()
    with ZipFile(memory_file, 'w') as zf:
        for i in range(len(content)):
            imgFile = content[i][1]['imagefile']
            txtFile = content[i][1]['textfile']
            zf.write(solvePath(imgFile), imgFile)
            zf.write(solvePath(txtFile), txtFile)
        zf.close()
    memory_file.seek(0)
    print("About to be sent!")
    return send_file(memory_file, as_attachment=True, download_name="file.zip", mimetype="application/zip")

#for some reason the dict turned into a list out of nowhere idk what is going on here
@app.route("/v1/post/epochcount", methods=['POST'])
def epochCount():
    print("Someone is reporting an epoch completition.")
    try:
        content = request.get_json(force=True)
    except Exception:
        return(Response("Failed decoding JSON", status=400))
    for i in range(len(content)):
        entryId = content[i][1]['entryId']
        currentNumOfEpoch = filesDict[int(entryId)]['epochs']
        newNumOfEpoch = currentNumOfEpoch + 1
        filesDict[entryId]['epochs'] = newNumOfEpoch
        filesDict[entryId]['assigned'] = False
        filesDict[entryId]['assignedExpirationDate'] = "none"
    print("Saved Successfully")
    return(Response(status=200))

statstest = {}

@app.route("/v1/post/stats", methods=['POST'])
def statspost():
    actualtime = gt()
    statstest[str(actualtime)] = request.get_json(force=True)
    print(statstest)
    return(Response(status=200))

pingtest = {}

@app.route("/v1/post/ping", methods=['POST'])
def pingpost():
    actualtime = gt()
    pingtest[str(actualtime)] = request.get_data(as_text=True)
    print(pingtest)
    return(Response(status=200))

#hardcode config cuz too tired
@app.route("/v1/get/config")
def getconf():
    #TODO: make it so its a omegaconf obj, conv to dict
    # receive the dict on local (peer) server, conv back to
    # omegaconf obj, and append to existing conf
    dict_with_configuration = {
        "model": "runwayml/stable-diffusion-v1-5",
        "extended_chunks": 2,
        "clip_penultimate": True,
        "fp16": True,
        "resolution": 512,
        "seed": 42,
        "train_text_encoder": True,
        "lr": "5e-6",
        "ucg": "0.1",
        "use_ema": False,
        "lr_scheduler": "cosine",
        # Advanced, do not touch
        "opt_betas_one": "0.9",
        "opt_betas_two": "0.999",
        "opt_epsilon": "1e-08",
        "opt_weight_decay": "1e-2",
        "buckets_shuffle": True,
        "buckets_side_min": "256",
        "buckets_side_max": "768",
        "lr_scheduler_warmup": "0.05" # Recheck this in the future if we get grad offloading with HM
    }
    return(jsonify(dict_with_configuration))

@app.route('/v1/get/lr_schel_conf')
def get_lr_conf():
    info = {
        "ImagesPerEpoch": str(get_num_rows()),
        "Epochs": "10",
    }
    return jsonify(info)

class BackgroundTasks(threading.Thread):
    def run(self,*args,**kwargs):
        print("Press q to quit background checker")
        while True:
            time.sleep(10/1000)
            actualTime = gt()
            for i in filesDict:
                expectedTime = filesDict[i]['assignedExpirationDate']
                if expectedTime != "none":
                    if actualTime > float(expectedTime):
                        entryId = filesDict[i]['entryId']
                        print("De-assigning entry " + str(entryId))
                        filesDict[i]['assigned'] = False
                        filesDict[i]['assignedExpirationDate'] = "none"


backgroundTask1 = BackgroundTasks()
backgroundTask1.start()

if __name__ == '__main__':
    app.run(debug=True, port=9090, host="0.0.0.0")
    
