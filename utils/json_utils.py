
import json
import subprocess
import time

import torch

directory = "/home/dcasals/TFG/"


def load_json_file(name,directory):
    try:
        with open(directory+name, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        raise FileNotFoundError
  


def load_json_model(name,directory):
    with open(directory+name,'r') as f:
        data_dict =  json.load(f)

    loaded_state_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, list):
            loaded_state_dict[key] = torch.tensor(value)
        else:
            loaded_state_dict[key] = torch.tensor(value)
    return loaded_state_dict


def save_json_file(data,name,directory):
    try:
        with open(directory+ name, 'x') as json_file:
            json.dump(data, json_file) 
    except FileExistsError:
        with open(directory+name, 'w') as json_file:
            json.dump(data, json_file) 
    

def save_json_model(model,name,directory):
    d = model.state_dict()
    serializable_dict = {}
    for key, value in d.items():
        if isinstance(value, list):
            serializable_dict[key] = value.tolist() # type: ignore
        else:
            serializable_dict[key] = value.numpy().tolist()
    save_json_filemodel(name,serializable_dict,directory)


def save_json_filemodel(name,data,directory):
    print("saving:", name)
    try:
        with open(directory+name, 'x') as json_file:
            json.dump(data, json_file) 
    except FileExistsError:
        with open(directory+name, 'w') as json_file:
            json.dump(data, json_file) 
    print(name , "saved.:")
    print(name, "model saved.")



def get_http_file(ip,origin,name,directory):
  print("status from: ",ip)
  curl_command = f"curl -o {directory}{name} https://{ip}/{origin}"
  print(curl_command)
  result = subprocess.run(curl_command, shell=True,capture_output=True)
  time.sleep(2)
  return result.returncode



def set_SWARN_http_initial_status(author,first,directory):
  aux = {
  "turno": first,
  "autor": author,
  'invalid':True,
  "info" : {}
}
  save_json_file(aux,"status_swarn.json",directory)
  save_json_file(aux,"status_swarn_before.json",directory)
  return aux

def set_next_round(nodes,directory):
    aux = {
    "general": "Ready",
    }
    for node in nodes:
        aux[node] = "Train"
    save_json_file(aux,"status_fed.json",directory)
    for node in nodes:
        save_json_file(aux,node+"_status_fed.json",directory)
        
    

def set_FED_initial_status(author,nodes,directory):

    aux = {
    "general": "Ready",
    }
    for node in nodes:
        aux[node] = "Train"
        print(aux)
    save_json_file(aux,"status_fed.json",directory)
    save_json_file(aux,"status_fed_before.json",directory)


def set_FED_ngrok_initial_status(nodes,directory):
    aux = {
    "general": "Ready",
    }
    print(nodes)
    for node in nodes:
        if node not in aux:
            aux[node] = {"info": {"state": ""}} # type: ignore
        
        # Asignar el estado "Train"
        aux[node]["info"]["state"] = "Train" # type: ignore
        print(aux)
    save_json_file(aux,"status_fed.json",directory)
    save_json_file(aux,"status_general_fed.json",directory)
    return aux


def set_FED_wormhole_initial_status(nodes,directory):
    aux = {
    "general": "Ready",
    }
    print(nodes)
    for node in nodes:
        if node not in aux:
            aux[node] = {"info": {"state": "Train", # type: ignore
                                  "code": ""}}
        
        # Asignar el estado "Train"
        aux[node]["info"]["state"] = "Train" # type: ignore
        print(aux)
    save_json_file(aux,"status_fed.json",directory)
    save_json_file(aux,"fed_general_status.json",directory)
    return aux
