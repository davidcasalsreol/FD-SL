import os
import random
import time
from typing import Any
import warnings
import torch

from utils.json_utils import get_http_file, load_json_file, load_json_model, save_json_file, save_json_model, set_FED_initial_status, set_FED_ngrok_initial_status, set_FED_wormhole_initial_status, set_next_round
from utils.model_utils import evaluate, load_pretrained_model, personalize_pretrained_model, save_pretrained_model, train_pretrained_model_summary
from utils.sftp_utils import send_sftp_file
from utils.wormhole_utils import get_all_models_wormhole, wormhole_receive_model, wormhole_send_model

def is_available_averaging(nodes,general,directory):
    print('MIROO')
    for node in nodes:
        if node != general:
            try:
                status = load_json_file(node+"_status_fed.json",directory)
                print("EY",node,status)
                print(status[node])
                if status[node] == "Train":
                    return False
            except FileNotFoundError:
                return False
    return True


def is_available_averaging_ngrok(nodes,directory):
    for node in nodes:
        try:
            status_file = node + "_status_fed.json"
            if not os.path.exists(directory +  status_file):
                print(f"File not found: {status_file}")
                return False
            print(os.path.exists(directory +  status_file))
            status = load_json_file(node+"_status_fed.json",directory)
            print("EY",node,status)
            print(status[node])
            if status[node]['info']['state'] != "Uploaded":
                return False
        except FileNotFoundError:
            return False
    return True

    


def averaging(model, nodes,directory):
    print("hellllo")
    modelo_general = model
    for key, value in modelo_general.state_dict().items():
        acc = torch.zeros_like(value)  # Initialize accumulator tensor
        for node in nodes:
            # modelo_local = CNN_Classification()
            modelo_local = load_pretrained_model('koko')
            modelo_local.load_state_dict(load_json_model(node + "_fed_local_model.json",directory))
            acc += modelo_local.state_dict()[key]  # Accumulate the tensor values
        acc /= len(nodes)  # Divide by the number of models (assuming n is the number of models)
        modelo_general.state_dict()[key] = acc
    return modelo_general


def averaging_ngrok(model_type, nodes):
    print("hellllo")
    model_avg = load_pretrained_model(model_type)
    n = len(nodes)
    k = 0
    for node in nodes:
        print(node)
        model = load_pretrained_model(model_type)
        model = personalize_pretrained_model(model_type,)
        model.load_state_dict(torch.load(directory+node+"_fed_local_model.pth")) # type: ignore
        with torch.no_grad():
            if k == 0: # primera iteracion
                for param , param_avg in zip(model.parameters(), model_avg.parameters()): # type: ignore
                    param_avg.data.copy_((param.data/n))
            else:
                for param , param_avg in zip(model.parameters(), model_avg.parameters()): # type: ignore
                    param_avg.data.copy_((param.data/n + param_avg.data))
    return model_avg

## esta funcion hace el average de todos los parametros del modelo, sean entrenables o no, quiero que solo haga el average de los entrenables en una nueva version llamada average_multinmodel
# donde igial que average_ngrok hara el average entre los modelos locales pero de la cpase model_type pero solo paramtreo entrenables
def averaging_multimodel(model_type, nodes,directory,class_names):
    print("hellllo")
    model_avg = load_pretrained_model(model_type)
    model_avg = personalize_pretrained_model(model_avg,class_names)
    n = len(nodes)
    k = 0
    for node in nodes:
        print(node)
        model = load_pretrained_model(model_type)
        model = personalize_pretrained_model(model,class_names)
        model.load_state_dict(torch.load(directory+node+"_fed_local_model.pth")) # type: ignore
        if k == 0: # primera iteracion
            for param , param_avg in zip(model.parameters(), model_avg.parameters()): # type: ignore
                if param.requires_grad:
                    param_avg.data.copy_((param.data/n))
                else: 
                    param_avg.data.copy_(param.data)
        else:
            for param , param_avg in zip(model.parameters(), model_avg.parameters()): # type: ignore
                if param.requires_grad:
                    param_avg.data.copy_((param.data/n + param_avg.data))
    return model_avg

def get_all_status(ips,nodes,directory):
    for node in nodes:
        get_http_file(ips[node],"status_fed.json",node+"_status_fed.json",directory)


def get_all_models(ips,nodes,directory):
    for node in nodes:
        get_http_file(ips[node],"fed_local_model.json",node+"_fed_local_model.json",directory)


def fit_fed_sftp(epochs, lr, model, train_loader,val_loader,name,general,nodes,working_dir,ips,opt_func = torch.optim.SGD, rounds=5):
    # "windows":"admin@172.16.2.91",
    acc_max = 0
    acc_max_gen = 0
    history = []
    optimizer = opt_func(model.parameters(),lr)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()
    # federated_models = [(i,model) for i in range(nodes)]
    set_FED_initial_status(name,nodes)  
    c = 0  
    for round in range(rounds):
        comunication_data = load_json_file("status_fed.json")
        ready = comunication_data["general"]  
        while ready != "Ready":
            c += 1
            time.sleep(10)
            print("esperando: ",c)
            comunication_data = load_json_file("status_fed.json")
            ready = comunication_data["general"]
            print(ready)
        model.load_state_dict(torch.load(working_dir+'fed_general_model.json'))
        comunication_data[general]= "Wait"
        comunication_data[name] = "Train"
        save_json_file(comunication_data,"status_fed.json")
        # modelo=load_model(CNN_model,"fed_best_model.pt")
        for epoch in range(epochs):
            t1 = time.time()    
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            acc = result['val_acc']
            st = '          '
            
            if acc>acc_max:
                epoch_max = epoch
                acc_max = acc
                st = '   ****    '
            t2 =  "{:6.1f} ".format(time.time() - t1)
            model.epoch_end(epoch, result,st+t2)
            history.append(result)  
            save_pretrained_model(model,"fed_local_model.json")
            
        comunication_data[name] = "Uploaded"
        print("rounds: ",round)
        if name != general:
            send_sftp_file(ips[general],"fed_local_model.json",name+"__fed_local_model.json")
            send_sftp_file(ips[general],"status_fed.json",name + "_status_fed.json")
        else:
            while not is_available_averaging(nodes,general):
                time.sleep(10)
            
            avg_model = averaging(model,nodes)
            # update_server_models(model,"fed",nodes)
            result = evaluate(avg_model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            acc = result['val_acc']
            st = '          '
            #guardando solo si mejora
            if acc>acc_max_gen:
                epoch_max = epoch
                accs_max = acc
                acc_max_gen = accs_max
                st = '   ***    '
                save_json_model(avg_model,"fed_general_model.json")
                print("modelo guardado") 
                print("UPDATEEEEE")
            for n in nodes:
                save_json_file(comunication_data,"status_fed.json")
                if n != general:
                    send_sftp_file(ips[n],'fed_general_model.json','fed_general_model.json')
                    send_sftp_file(ips[n],"fed_status.json","fed_status.json")
            

    print(rs)
    torch.save(model.state_dict(),'fed_last_model.pt')
    print("Best model saved best_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc_max,epoch_max))
    print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    rta = {
        "history": history,
        "model":"fed_general_model.json"
    }
    return rta



def fit_fed_ngrok(epochs, lr, model, train_loader,val_loader,name,general,nodes, ips,model_type,opt_func = torch.optim.SGD, rounds=5):
    # "windows":"admin@172.16.2.91",
    acc_max = 0
    acc_max_gen = 0
    history = []
    optimizer = opt_func(model.parameters(),lr)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()
    # federated_models = [(i,model) for i in range(nodes)]
    print(name)
    set_FED_ngrok_initial_status(nodes)  
    c = 0  
    for round in range(rounds):
        comunication_data = load_json_file("status_fed.json")
        ready = comunication_data["general"]  
        while ready != "Ready":
            c += 1
            time.sleep(10)
            print("esperando: ",c)
            get_http_file(ips[general],"status_fed.json","fed_general_status.json")
            comunication_data = load_json_file("fed_general_status.json")
            ready = comunication_data["general"]
            print(ready)
            if ready:
                print(comunication_data[name]["info"])
                get_http_file(ips[general],"fed_general_model.json","fed_general_model.json")
        if name == general:
            comunication_data["general"] = "Wait"
            save_json_file(comunication_data,"fed_general_status.json")
        else: 
            print("waiting general to update general status")
            time.sleep(3)
        model.load_state_dict(load_json_model('fed_general_model.json'))
        comunication_data["general"]= "Wait"
        save_json_file(comunication_data,"status_fed.json")
        # modelo=load_model(CNN_model,"fed_best_model.pt")
        for epoch in range(epochs):
            t1 = time.time()    
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            acc = result['val_acc']
            st = '          '
            
            if acc>acc_max:
                epoch_max = epoch
                acc_max = acc
                st = '   ****    '
            t2 =  "{:6.1f} ".format(time.time() - t1)
            model.epoch_end(epoch, result,st+t2)
            history.append(result)  
            save_json_model(model,"fed_local_model.json")
        
        comunication_data[name]["info"]["state"] = "Uploaded"
        save_json_file(comunication_data,"status_fed.json")
        print("rounds: ",round)
        if name == general:
            get_all_status(ips,nodes)
            while not is_available_averaging_ngrok(nodes):
                time.sleep(10)
                get_all_status(ips,nodes)
            get_all_models(ips,nodes)
        
            avg_model = averaging_ngrok(model_type,nodes)
            # update_server_models(model,"fed",nodes)
            result = evaluate(avg_model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            acc = result['val_acc']
            st = '          '
            #guardando solo si mejora
            if acc>acc_max_gen:
                epoch_max = epoch
                accs_max = acc
                acc_max_gen = accs_max
                st = '   ***    '
                save_json_model(avg_model,"fed_general_model.json")
                print("modelo guardado") 
                print("UPDATEEEEE")
            else: 
                print("\n\n\n SE PARTE DESDE EL ANTERIOR MODELO GENERAL\n\n\n")
            comunication_data = set_FED_ngrok_initial_status(nodes)
            save_json_file(comunication_data,"fed_general_status.json")
        
            

    print(rs)
    torch.save(model.state_dict(),'fed_last_model.pth')
    print("Best model saved best_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc_max,epoch_max))
    print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))

    rta = {
        "history": history,
        "model":"fed_general_model.json"
    }
    return rta



def fit_fed_wormhole(epochs, lr, model, train_loader,val_loader,name,general,nodes, ips,model_type,opt_func = torch.optim.SGD, rounds=5):
    # "windows":"admin@172.16.2.91",
    acc_max = 0
    acc_max_gen = 0
    history = []
    optimizer = opt_func(model.parameters(),lr)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()
    # federated_models = [(i,model) for i in range(nodes)]
    set_FED_wormhole_initial_status(nodes)  
    c = 0  
    for round in range(rounds):
        get_http_file(ips[general],"fed_general_status.json","fed_general_status.json")
        comunication_data = load_json_file("fed_general_status.json")
        ready = comunication_data["general"]  
        while ready != "Ready":
            c += 1
            time.sleep(10)
            print("esperando: ",c,"\n\n\n       NUEVA RONDA           \n\n\n")
            get_http_file(ips[general],"status_fed.json","fed_general_status.json")
            comunication_data = load_json_file("fed_general_status.json")
            ready = comunication_data["general"]
            print(ready)
            if ready == 'Ready' and round != 0:
                print('000000000000',comunication_data[name]["info"])
                wormhole_receive_model(comunication_data[name]["info"]['code'],"fed_general_model.json")
        if name == general:
            print('1111111')
            comunication_data["general"] = "Wait"
            save_json_file(comunication_data,"fed_general_status.json")
            print("waiting general to update general status")
        while ready == "Ready":
            print('222222222')
            get_http_file(ips[general],"fed_general_status.json","status_fed.json")
            comunication_data = load_json_file("status_fed.json")
            ready = comunication_data["general"]
            time.sleep(1)
        model.load_state_dict(load_json_model('fed_general_model.json'))
        # modelo=load_model(CNN_model,"fed_best_model.pt")
        for epoch in range(epochs):
            t1 = time.time()    
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            acc = result['val_acc']
            st = '          '
            
            if acc>acc_max:
                epoch_max = epoch
                acc_max = acc
                st = '   ****    '
            t2 =  "{:6.1f} ".format(time.time() - t1)
            model.epoch_end(epoch, result,st+t2)
            history.append(result)  
            save_json_model(model,"fed_local_model.json")
        
        comunication_data[name]["info"]["state"] = "Uploaded"
        phrase = wormhole_send_model('fed_local_model.json')
        comunication_data[name]["info"]["code"] = phrase
        save_json_file(comunication_data,"status_fed.json")
        print("rounds: ",round, phrase)
        if name == general:
            get_all_status(ips,nodes)
            t = is_available_averaging_ngrok(nodes)
            print(t)
            while not is_available_averaging_ngrok(nodes):
                print('2waittitititititing')
                time.sleep(10)
                get_all_status(ips,nodes)

            get_all_models_wormhole(nodes)
            print('\n\n\n\n\n evaluatinn \n\n\n\n\n ')
            avg_model = averaging_ngrok(model_type,nodes)
            # update_server_models(model,"fed",nodes)
            result = evaluate(avg_model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            acc = result['val_acc']
            st = '          '
            #guardando solo si mejora
            if acc>acc_max_gen:
                epoch_max = epoch
                accs_max = acc
                acc_max_gen = accs_max
                st = '   ***    '
                save_json_model(avg_model,"fed_general_model.json")
                print("modelo guardado") 
                print("UPDATEEEEE")
            else: 
                print("\n\n\n SE PARTE DESDE EL ANTERIOR MODELO GENERAL\n\n\n")
            
            comunication_data["general"] = "Ready"
            print("ENVOANDOOO A LOCALLEEEES","\n\n\n       NUEVA RONDA           \n\n\n")
            for node in nodes:
                phrase = wormhole_send_model("fed_general_model.json")
                comunication_data[node]["info"]["state"] = "Train"
                comunication_data[node]["info"]["code"] = phrase
            save_json_file(comunication_data,"fed_general_status.json")

        


            

    print(rs)
    # torch.save(model.state_dict(),"fed_general_status.json")
    print("Best model saved best_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc_max,epoch_max))
    print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    rta = {
        "history": history,
        "model":"fed_general_model.json"
    }
    return rta





def fit_fed_multimodel_sftp(rounds,epochs, optimizer, model, train_loader,val_loader,name,general,nodes, ips,working_dir,working_dirs,opt_lr,model_type,writer,data_name,loss_fn,early_stop):
    # "windows":"admin@172.16.2.91",
    acc_max = 0
    acc_max_gen = 0
    history = []
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()
    # federated_models = [(i,model) for i in range(nodes)]
    set_FED_initial_status(name,nodes,working_dir)
    c = 0
    for round in range(rounds):
        comunication_data = load_json_file("status_fed.json",working_dir)
        ready = comunication_data["general"]
        while ready != "Ready":
            c += 1
            time.sleep(10)
            print("esperando: ",c)
            comunication_data = load_json_file("status_fed.json",working_dir)
            ready = comunication_data["general"]
            print(ready)
        if name == general:
            time.sleep(10)
            "WAITINGGG THEM TO STARTTT"
        model.load_state_dict(torch.load(working_dir+'fed_general_model.pth'))
        comunication_data["general"]= "Wait"
        comunication_data[name] = "Train"
        save_json_file(comunication_data,"status_fed.json",working_dir)
        # modelo=load_model(CNN_model,"fed_best_model.pt")
        print(writer)
        results,acc_max,ep_max= train_pretrained_model_summary(epochs=epochs,model= model,train_loader= train_loader,val_loader= val_loader,name=model_type,path=working_dir,acc_max=acc_max,type='fed',rounds=rounds,writer=writer,loss_fn=loss_fn,opt_func=optimizer,earlystop=early_stop)
        comunication_data[name] = "Uploaded"
        save_json_file(comunication_data,"status_fed.json",working_dir)

        print("rounds: ",round)
        history.append(results)
        send_sftp_file(ips[general],working_dir+"fed_local_model.pth",working_dirs[general]+name+"_fed_local_model.pth")
        send_sftp_file(ips[general],working_dir+"status_fed.json",working_dirs[general] +name + "_status_fed.json")
        if name != general:    
            print('sleeping')
            time.sleep(20)
        else:
            while not is_available_averaging(nodes,general,working_dir):
                time.sleep(10)
            time.sleep(10)
            print("we are averaging")
            avg_model = averaging_multimodel(model_type=model_type,nodes=nodes,directory=working_dir,class_names=train_loader.dataset.classes)

            # update_server_models(model,"fed",nodes)
            result = evaluate(avg_model, val_loader)
            acc = result['val_acc']
            st = '          '
            #guardando solo si mejora
            if acc>acc_max_gen:
                epoch_max = round
                accs_max = acc
                acc_max_gen = accs_max
                st = '   ***    '
                save_pretrained_model(avg_model,working_dir,'fed_general_model.pth')
                print("modelo mejorado")
                print("UPDATEEEEE")
            set_next_round(nodes,working_dir)
            for n in nodes:
                if n != general:
                    send_sftp_file(ips[n],working_dir+'fed_general_model.pth',working_dirs[n]+'fed_general_model.pth')
                    send_sftp_file(ips[n],working_dir+"status_fed.json",working_dirs[n]+"status_fed.json")
    
        writer.add_scalars(main_tag="Round_Accuracy", 
                        tag_scalar_dict={"round_acc": acc_max}, 
                        global_step=round)
        writer.close() 
        # Close the writer
    timestamp = time.strftime("%Y-%m-%d-%H")
    writer_path = f'{working_dir}/results/{data_name}/fed/{timestamp}/{len(list(ips.keys()))}_devices_{rounds}_rounds_{epochs}_epochs/'
    save_pretrained_model(model,writer_path,f'{len(list(ips.keys()))}_num_devices_{rounds}_rounds_{epochs}_epochs_{opt_lr}_optimizer_last.pth')
    model.load_state_dict(torch.load(working_dir+'fed_general_model.pth'))
    save_pretrained_model(model,writer_path,f'{len(list(ips.keys()))}_num_devices_{rounds}_rounds_{epochs}_epochs_{opt_lr}_optimizer_general.pth')



        
    print(rs)
    save_pretrained_model(model,working_dir,"fed_general_model.pth")

    #print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    rta = {
        "history": history,
        "model":"fed_general_model.json"
    }
    return rta, acc_max_gen, ep_max


