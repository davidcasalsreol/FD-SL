import json
import os
from flask import Flask, jsonify, send_from_directory
import torch
import requests


app = Flask(__name__)

# Cambia el directorio de trabajo según el sistema operativo
if os.name == 'nt':  # Windows
    os.chdir('C:\\Temp')
elif os.name == 'posix':  # Linux/macOS y otros sistemas tipo Unix
    os.chdir('/tmp')

# Ruta desde donde se publicará el servicio y puerto en el que escuchará
HOST = '0.0.0.0'
PORT = 81

if os.name == 'nt':
    file_directory = 'C:\\Temp'
else:
    file_directory = '/home/dcasals/TFG/'


@app.route('/')
def index():
    files = os.listdir(file_directory)
    return jsonify(files)


@app.route('/<path:filename>',methods=['GET'])
def serve_file(filename):
    print("returnininining", file_directory, filename)
    f = send_from_directory(file_directory, filename)
    print(f)
    return f

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
