
import json
import os
from pathlib import Path
import subprocess
import time

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from utils.DataManager import create_dataloaders
from utils.fed_train import fit_fed_multimodel_sftp
from utils.results_utils import compress_folders, interactive_visualizer
from utils.json_utils import load_json_file, save_json_file
from utils.sftp_utils import send_sftp_file
from utils.swarn_train import fit_swarn_multimodel_sftp
from utils.writer_utils import create_summary_writer
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from utils.model_utils import get_labels, get_prediction, load_pretrained_model, personalize_pretrained_model, save_pretrained_model
from sklearn.metrics import confusion_matrix


import torch.optim as optim

def seleccionar_optimizador(numero):
    if numero == 1:
        return optim.SGD
    elif numero == 2:
        return optim.Adam
    elif numero == 3:
        return optim.RMSprop
    elif numero == 4:
        return optim.Adagrad
    elif numero == 5:
        return optim.AdamW
    elif numero == 6:
        return optim.Adamax
    elif numero == 7:
        return optim.Adadelta
    elif numero == 8:
        return optim.SparseAdam
    elif numero == 9:
        return optim.LBFGS
    else:
        raise ValueError("Número fuera de rango. Debe ser un valor del 1 al 9.")

def get_input(prompt):
    """Gets input from the user."""
    while True:
        try:
            value = input(prompt)
            return value
        except ValueError:
            print("Invalid input. Please enter a valid value.")

def wait_start(ips,experiment_number,working_paths,id):
    nodes = list(ips.keys())
    for node in nodes:
        if node != id:
            try:
                print(f"experiments/{node}_experiment.json",working_paths[id])
                f = load_json_file(f"experiments/{node}_experiment.json",working_paths[id])
            except FileNotFoundError:
                return False
            if f["experiment"] != experiment_number:
                print(f["experiment"],experiment_number)
                return False
    return True

def summary_executioner(light,train_dataloader,val_dataloader,test_dataloader,models,types,rounds, epochs,optimizer,working_paths,id,ips,class_names,data_name,o,lr,loss_fn = nn.CrossEntropyLoss(),earlystop = 5):

    experiment_number = 0
    timestamp = time.strftime("%Y-%m-%d-%H")
    acc_max = dict()
    experiment_path = Path(working_paths[id]+f'experiments/Fed')
    experiment_path.mkdir(parents=True,
                        exist_ok=True)

    experiment_path = Path(working_paths[id]+f'experiments/Swarn')
    experiment_path.mkdir(parents=True,
                        exist_ok=True)
    output_zip = Path(working_paths[id]+f'experiments/{timestamp}_execution_experiment')
    zip_files = []
    zip_files.append(f'experiments/{timestamp}_summary.json')
    # el diccionario de accmax comxo primera key tenndra el tiop entrenamiento y dentro tendra un diccionario donde la key sera el modeloy el valor la acc_maxima hasta el momento
    for t in types:
        acc_max.setdefault(t, {})
        for m in models:
            acc_max[t].setdefault(m, 0.0)
    for model_name in models:
        for t in types: 
            cc = 0
            for opt_fun in optimizer:
                for l in lr:
                    pretrained_model = load_pretrained_model(model_name_str=model_name,light=light)
                    model = personalize_pretrained_model(model=pretrained_model,class_names=class_names)
                    opt_lr = o[cc] +'_' +str(l)

                    print(opt_lr)
                    opt = opt_fun(model.parameters(), lr=l) # type: ignore
                    for r in rounds:    
                        for epoch in epochs: 
                            experiment_number += 1
                            nodes = list(ips.keys())
                            data = {"experiment": experiment_number}
                            file_model_name = f"{model_name}_{len(list(ips.keys()))}_devices__{r}_rounds_{epoch}_epochs_{opt_lr}.pth"
                            writer_path =f'results/{data_name}/{t}/{timestamp}/{file_model_name}/'
                            writer = create_summary_writer(working_paths[id]+writer_path)
                            zip_files.append(writer_path)
                            save_json_file(data,f"experiments/{id}_experiment.json",working_paths[id])
                            for node in nodes:
                                if node != id:
                                    send_sftp_file(ips[node],f"{working_paths[id]}experiments/{id}_experiment.json", f"{working_paths[node]}experiments/{id}_experiment.json")
                            start = wait_start(ips,experiment_number,working_paths,id)
                            while not start:
                                start = wait_start(ips,experiment_number,working_paths,id)
                                time.sleep(30)
                            print(f"[INFO] Experiment number: {experiment_number}")
                            print(f"[INFO] Model: {model_name}")
                            print(f"[INFO] DataLoader: Pecas")
                            print(f"[INFO] Number of epochs: {epochs}") 
                            if t == "Swarn":
                                save_pretrained_model(model=model,target_dir=working_paths[id],model_name="swarn_best_model.pth")
                                try:
                                    next = int(id) + 1 if int(id) + 1 < len(working_paths) else 0
                                except ValueError:
                                    ids = list(ips.keys())
                                    id_index = ids.index(id)
                                    next = ids[id_index + 1] if id_index + 1 < len(ids) else ids[0]
                                h, acc, ep = fit_swarn_multimodel_sftp(model=model,train_loader=train_dataloader,val_loader=val_dataloader,rounds=r,epochs=epoch,model_type=model_name,name=id,first=list(ips.keys())[0],ips=ips,working_dir=working_paths[id],next_working_dir=working_paths[next],data_name=data_name,writer = writer,optimizer=opt,opt_lr=opt_lr,loss_fn=loss_fn)
                                model.load_state_dict(torch.load(working_paths[id]+'swarn_best_model.pth')) # type: ignore
                                
                            elif t == "Fed":
                                save_pretrained_model(model=model,target_dir=working_paths[id],model_name="fed_general_model.pth")
                                h, acc, ep= fit_fed_multimodel_sftp(model=model,train_loader=train_dataloader,val_loader=val_dataloader,rounds=r,epochs=epoch,name=id,general=list(ips.keys())[0],model_type=model_name,ips =ips,working_dirs=working_paths,working_dir=working_paths[id],nodes=ips.keys(),data_name=data_name,writer=writer,optimizer=opt,opt_lr=opt_lr,loss_fn=loss_fn,early_stop=earlystop)
                                model.load_state_dict(torch.load(working_paths[id]+'fed_general_model.pth')) # type: ignore

                            save_pretrained_model(model=model,
                                        target_dir=working_paths[id]+f"/models/{data_name}/{t}/",
                                        model_name=file_model_name)
                            print("*"*50 + "\n")

                            ##MAKE TEST ACCURACY + CONFUSION MATRIX && SAVE IT
                            ytest = get_labels(model,test_dataloader)
                            ypred = get_prediction(model,test_dataloader)
                            
                            # Calculate confusion matrix

                            acc   = accuracy_score(ytest,ypred)
                            print(f"Accuracy: {acc}")
                            C     = confusion_matrix(ytest,ypred)
                            # Plot confusion matrix
                            plt.figure(figsize=(10, 8),)
                            plt.imshow(C, cmap=plt.cm.Blues)
                            plt.title("Confusion Matrix")
                            plt.colorbar()
                            plt.xlabel("Predicted Label")
                            plt.ylabel("True Label")
                            plt.xticks(range(len(class_names)), class_names, rotation=90)
                            plt.yticks(range(len(class_names)), class_names)
                            plt.show()

                            item = {"experiment": experiment_number,
                                    "path_to_model":f"models/{data_name}/{t}/"+file_model_name,
                                    "model": model_name,
                                    "type": t,
                                    "rounds": r,
                                    "epochs": epoch,
                                    "test_accuracy": acc,
                                    "confusion_matrix": 'None',
                                    "writer_logs": writer_path,
                                    # "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
                                    }
                            #add to priority quueue with results_utils
                            try:
                                with open(working_paths[id]+f"experiments/{timestamp}_summary.json", 'r') as f:
                                    data = json.load(f)
                                    items_list = data.get('items', [])
                            except FileNotFoundError:
                                with open(working_paths[id]+f"experiments/{timestamp}_summary.json", 'x') as f:
                                    data = {"items": []}
                                    items_list = data.get('items', [])
                                    json.dump(data, f)
                            # Agregar nuevos elementos a la lista existente
                            items_list.append(item)
                            data['items'] = items_list
                            with open(working_paths[id]+f"experiments/{timestamp}_summary.json", 'w') as f:
                                json.dump(data,f)
                            zip_files.append(working_paths[id]+f"models/{data_name}/{t}/"+file_model_name)                                     
                cc +=1

    all_paths = ' '.join(zip_files)
    print(all_paths)
    prefix = working_paths[id]
     

    zip_coomand = f'zip -r {output_zip}.zip {all_paths}'
    subprocess.run(zip_coomand, shell=True)


def get_optimizers():
    print("ALL NODES SHOLD HAVE THIS STRUCTURE IDENTICAL")
    optimizer_dict = {
        1: "SGD",
        2: "Adam",
        3: "RMSprop",
        4: "Adagrad",
        5: "AdamW",
        6: "Adamax",
        7: "Adadelta",
        8: "SparseAdam",
        9: "LBFGS"
    }
    print('1. SGD\n2. Adam\n3. RMSprop\n4. Adagrad\n5. AdamW\n6. Adamax\n7. Adadelta\n8. SparseAdam\n9. LBFGS')
    opt = get_input("Enter the optimizers 1-9 separated by ',' (default Adam): ") or '2'
    opt = opt.split(",")
    optimizer = []
    oo = []
    for o in opt:
        optimizer.append(seleccionar_optimizador(int(o)))
        oo.append(optimizer_dict[int(o)])
    return optimizer, oo

def get_epochs():
    epochs = get_input("Enter the number of epochs [numbers separatedby comas] (default [15,20,30]): ") or '15, 20, 30'
    epochs = epochs.split(",")
    epochs = [int(e) for e in epochs]
    return epochs

def get_rounds():
    print("ALL NODES SHOLD HAVE THIS STRUCTURE IDENTICAL")
    rounds = get_input("Enter rounds separated by ',' (default [3, 5, 7] ): ") or '3, 5, 7'
    rounds = rounds.split(",")
    rounds = [int(r) for r in rounds]
    return rounds

def get_types():  
    print("ALL NODES SHOLD HAVE THIS STRUCTURE IDENTICAL")
    types = get_input("Enter the type of training (default ['Swarn', 'Fed']): ") or 'Swarn,Fed'
    types = types.split(",")
    types = [t for t in types]
    return types

def get_lr():
    lr = get_input("Enter lr separated by comas if several (default 0.001): ") or '0.001'
    lr = lr.split(",")
    lr = [float(l) for l in lr]
    return lr

def get_models():
    model_dict = {
    '1': "AlexNet",
    '2': "VGG16",
    '3': "VGG19",
    '4': "ResNet",
    '5': "ResNet34",
    '6': "ResNet50",
    '8': "DenseNet",
    '9': "Inception v3",
    '10': "GoogLeNet",
    '11': "ShuffleNet v2",
    '12': "MobileNetV2",
    '13': "MobileNetV3",
    '14': "ResNeXt",
    '15': "Wide ResNet 50",
    '16': "Wide ResNet 101",
    '17': "MNASNet",
    '18': "EfficientNet_B0",
    '19': "EfficientNet_B1",
    '20': "EfficientNet_B2",
    '21': "EfficientNet_Light",
    '22': "EfficientNet_V2_S",
    '23': "EfficientNet_V2_M",
    '24': "EfficientNet_V2_L",
}
    print("ALL NODES SHOLD HAVE THIS STRUCTURE IDENTICAL")
    print('1. AlexNet\n2. VGG16\n3. VGG19\n4. ResNet\n5. ResNet34\n6. ResNet50\n7. SqueezeNet\n8. DenseNet\n9. Inception v3\n10. GoogLeNet\n11. ShuffleNet v2\n12. MobileNetV2\n13. MobileNetV3\n14. ResNeXt\n15. Wide ResNet 50\n16. Wide ResNet 101\n17. MNASNet\n18. EfficientNet_B0\n19. EfficientNet_B1\n20. EfficientNet_B2\n21. EfficientNet_Light\n22. EfficientNet_V2_S\n23. EfficientNet_V2_M\n24. EfficientNet_V2_L')
    models = get_input("Enter pretrained models to use [numbers separated by coma](default [GoogLeNet,ResNet50]): ") or '10,4'
    models = models.split(",")
    models = [model_dict[m] for m in models]
    return models

def get_working_paths(devices):
    print("ALL NODES SHOLD HAVE THIS STRUCTURE IDENTICAL")
    base_working_path = {
        '1': '/home/dcasals/TFG/',
        '2': '/home/dcasals/TFG/',
        '3': '/Users/admin/TFG/',
        '4': '/Users/david/TFG/',

    }
    working_paths = []
    ok = get_input("Do you want to use the default local working paths? (Y/N): ") or 'N'
    if ok == "Y":
        for device in devices:
            working_paths.append(base_working_path['3'])
            working_paths.append(base_working_path['4'])
        return working_paths
    ok = get_input("Do you want to use the default remote workiing paths? (Y/N): ") or 'N'
    if ok == "Y":
        for device in devices:
            working_paths.append(base_working_path['1'])
            working_paths.append(base_working_path['2'])
        return working_paths
    for device in devices:
        working_path = get_input(f"Enter the working directories for device {device}: ")
        working_paths.append(working_path)
    return working_paths

def get_urls(num_devices):
    base_url = {
        "1": 'dcasals@kimun.ing.puc.cl',
        "2": 'dcasals@kintun.ing.puc.cl',
        "3": 'admin@127.0.0.1',
        "4": 'david@127.0.0.1',
    }
    urls = []
    print("ALL NODES SHOLD HAVE THIS STRUCTURE IDENTICAL")
    ok = get_input("Do you want to use the default local nodes? (Y/N): ")
    if ok == "Y":
        urls.append(base_url['3'])
        urls.append(base_url['4'])
        return urls
    ok = get_input("Do you want to use the default remote nodes? (Y/N): ")
    if ok == "Y":
        urls.append(base_url['1'])
        urls.append(base_url['2'])
        return urls
    for device in num_devices:
        url = get_input(f"Enter the url of the device {device}:")
        urls.append(url)
    return urls

def get_own_id():
    id = get_input("Enter the id YOUR (remember you will use the work_path with your ID given earlier): ")
    return id

def get_ids(num_devices):
    ids = [int(i) for i in range(num_devices)]
    print("this are default ids",ids)
    ok = get_input("Do you want to use numerical identifiers for the devices? (Y/N): ") or "N"
    if ok == "N": 
        ids = []
        for device in range(num_devices):
            id = get_input(f"Enter the id of the device {device}: ")
            ids.append(id)
        given = False
        while not given:
            id = get_input("Enter the id YOUR (remember you will use the work_path with your ID given earlier): ")
            if id in ids:
                return ids, id
            else:
                print("The id is not in the list")
    given = False
    while not given:
        id = int(get_input("Enter the id YOUR (remember you will use the work_path with your ID given earlier): "))
        if id in ids:
            return ids, id
        else:
            given = True

def get_loss_fn():
    dict = {
        1: nn.CrossEntropyLoss(),
        2: nn.MSELoss(),
        3: nn.L1Loss(),
        4: nn.BCELoss(),
        5: nn.BCEWithLogitsLoss(),
        6: nn.NLLLoss(),
        7: nn.KLDivLoss(),
    }
    print("1. nn.CrossEntropyLoss\n2. nn.MSELoss\n3. nn.L1Loss\n4. nn.BCELoss\n5. nn.BCEWithLogitsLoss\n6. nn.NLLLoss\n7. nn.KLDivLoss")
    num = int(get_input("Enter the number of the loss function (default 1): ") or 1)
    return dict[num]

def main():
    """
    This is the main function that executes the program.
    It prompts the user for various inputs, performs data processing,
    and calls other functions to execute the program logic.
    """
    num_devices = int(get_input("Enter the number of devices (default 2): ") or 2)
    ids,id = get_ids(num_devices) # type: ignore
    print(id)
    urls = get_urls(ids)
    ips = dict(zip(ids, urls))
    print(ips)
    models = get_models()
    types = get_types()
    lr = get_lr()  
    rounds = get_rounds()
    earlystop = int(get_input("Enter the number of earlystoppage (default 5): ") or 5)
    epochs = get_epochs()
    optimizer, o = get_optimizers()
    loss_fn = get_loss_fn()
    default = str(os.path.dirname(__file__))
    working_paths = get_working_paths(ids)
    working_paths_dict = dict(zip(ids, working_paths))
    dataset = get_input(f"Enter the dataset folder path (default is: {default+'/datasets/'}): ") or default + "/datasets/"

    list = os.listdir(dataset)
    print(f"Choose the dataset from the list: {list}")
    given = False
    while not given:
        data = get_input("Enter the dataset: ")
        if data in list:
            given = True
        else:
            print("The dataset is not in the list")
    width = int(get_input("Enter the width size (default 255): ") or 255)
    height = int(get_input("Enter the width size (default 255): ") or 255)
    batch_size = int((get_input("Enter the batch size (default 32): ")) or 32)

    train_path = dataset + data + "/train"
    val_path = dataset + data + "/val"
    test_path = dataset + data + "/test"
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    normalize
])   
    train_dataloader, val_dataloader, class_names = create_dataloaders(train_dir=train_path,
                                                                        test_dir=val_path,
                                                                        transform=transforms,
                                                                        batch_size=batch_size)

    num_workers = os.cpu_count() or 1
    type = ["Fed","Swarn"]
    test_set= ImageFolder(test_path, transform=transforms)
    test_data  = DataLoader(
        test_set,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        )
    working_paths_dict = dict(zip(ids, working_paths))
    print("modelos",models)
    print("tipos",types)
    print("rounds",rounds)
    print("epochs",epochs)
    print("optimizers",optimizer)
    print("lr",lr)
    print("loss_fn",loss_fn)
    print("working_paths",working_paths)
    print("id",id, ids)
    print("ips",ips)
    ok = get_input("Do you want to start the training? (Y/N): ") or 'Y'
    if ok != "Y":
        print('Change:\n 1. models\n 2. types\n 3. rounds\n 4. epochs\n 5. optimizer\n 6. lr\n 7. loss_fn\n 8. working_paths\n 9. id\n 10. ips')
        option = get_input("Enter the number of the option you want to change: ") or 'N'
        if option == "1":
            models = get_models()
        elif option == "2":
            types = get_types()
        elif option == "3":
            rounds = get_rounds()
        elif option == "4":
            epochs = get_epochs()
        elif option == "5":
            optimizer, o = get_optimizers()
        elif option == "6":
            lr = get_lr()
        elif option == "7":
            loss_fn = get_input("Enter the loss function (default nn.CrossEntropyLoss): ") or nn.CrossEntropyLoss
            loss_fn = nn.CrossEntropyLoss()
        elif option == "9":
            ids,id = get_ids(num_devices)
            working_paths = get_working_paths(ids)
            working_paths_dict = dict(zip(ids, working_paths))
        elif option == "8":
            working_paths = get_working_paths(ids)
            working_paths_dict = dict(zip(ids, working_paths))
        elif option == "10":
            urls = get_urls(ids)
            ips = dict(zip(ids, urls))
        else:
            print("Invalid option")
            return
    light = get_input("Do you want to use the light version of the model weights or the latest? (Y/N): ") or 'N'
    if light == "N":
        light = False
    else:
        light = True
    summary_executioner(light = light,train_dataloader=train_dataloader,val_dataloader= val_dataloader,test_dataloader=test_set,models= models, types=types, rounds=rounds,ips=ips,working_paths=working_paths_dict, id = id,epochs=epochs ,optimizer=optimizer,data_name = data,class_names=class_names,o=o,lr=lr, loss_fn=loss_fn) 

    models_data = interactive_visualizer()

    for data in models_data:
        print(data['path_to_model'])


if __name__ == "__main__":
    main()
