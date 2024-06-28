import random
import time
import warnings
import torch

from utils.model_utils import *
from utils.sftp_utils import *
from utils.web_utils import *
from utils.json_utils import *
from utils.wormhole_utils import  wormhole_receive_model, wormhole_send_model
from utils.writer_utils import create_summary_writer

# Definir la función para filtrar las advertencias
def filter_warning(message, category, filename, lineno, file=None, line=None):
    if "Found Intel OpenMP" in str(message):
        return None  # Ignorar la advertencia
    else:
        return True  # Continuar mostrando otras advertencias

# Aplicar el filtro de advertencias
    # warnings.showwarning = filter_warning


def fit_swarn(epochs, lr,model, data, opt_func = torch.optim.SGD,nodes = 2, earlystop=3):
    acc_max = 0
    history = []
    optimizer = opt_func(model.parameters(),lr)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    node_models =[]
    for node in range(nodes):
        node_models.append(model)
    t0 = time.time()
    for epoch in range(epochs):
        print(epoch)
        for node in range(nodes):
            # model.load_state_dict(torch.load('save_json_model.pt'))
            model.load_state_dict(load_json_model(name='swarn_json_model.json'))
            train_loader, val_loader = data[node]
            t1 = time.time()
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            print(1)
            result = evaluate(model, val_loader)
            print(2)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            print(3)
            acc = result['val_acc']
            st = '          '
            if acc>acc_max:
                epoch_max = epoch
                acc_max = acc
                st = '   ***    '
                # torch.save(model.state_dict(),'swarn_json_model.pt')
                save_json_model(model= model,name='swarn_json_model.json')
            t2 =  "{:6.1f} ".format(time.time() - t1)
            model.epoch_end(epoch, result,st+t2)
            history.append(result)
            print(epoch)
            if epoch - epoch_max >= earlystop:
                print(rs)
                print('*** Early stop after '+str(earlystop)+' epochs with no improvement')
                return history

    print(rs)
    torch.save(model.state_dict(),'swarn_last_model.pt')
    print("Best model saved best_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc_max,epoch_max))
    print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    return history


def fit_swarn_wormhole(epochs, lr,model, train_loader,val_loader, name,next_server,before_server,ips, opt_func = torch.optim.SGD, earlystop=5,rounds = 5):
    acc_max = 0
    sendings = 0
    receivings = 299
    random.seed(2)
    aleatorios = [16777217 for _ in range(300)]
    worm_code_send = aleatorios[sendings]
    worm_code_receive = aleatorios[receivings]
    history = []
    optimizer = opt_func(model.parameters(),lr)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()
    comunication_data = set_SWARN_http_initial_status(name,0)
    c = 0
    for round in range(rounds):
        comunication_data = load_json_file("status_swarn.json")
        turno = comunication_data["turno"]
        autor = comunication_data["autor"]
        invalid = comunication_data["invalid"]
        print("r:", round,comunication_data)
        while turno != autor and not invalid:
            c += 1
            time.sleep(8)
            get_http_file(ips[next_server],"status_swarn.json","status_swarn_before.json")
            comunication_data = load_json_file("status_swarn.json")
            turno = comunication_data["turno"]
            autor = comunication_data["autor"]
            invalid = comunication_data["invalid"]
            print("esperando: ",c,comunication_data)

        if round == 0 and name == "apple":
            comunication_data["invalid"] = False
        else:
            r = comunication_data['info']['receivings']
            worm_code_receive = aleatorios[r]
            wormhole_receive_model(worm_code_receive,"swarn_json_model.json")

        model.load_state_dict(load_json_model("swarn_json_model.json"))
        # model.load_state_dict(torch.load('swarn_best_model.pt'))
        

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
                st = '   ***    '
            save_json_model(model,"swarn_json_model.json") ##MIRAR CON TORCH.SAVE
            # torch.save(model.state_dict(),'swarn_best_model.pt')
            print("AAAAAAA")
            # json_data = convert_model_to_json(model)
            # with open('swarn_json_model.json', 'w') as json_file:
            #     json_file.write(json_data)
            t2 =  "{:6.1f} ".format(time.time() - t1)
            model.epoch_end(epoch, result,st+t2)
            history.append(result)
            if epoch - epoch_max >= earlystop:
                print(rs)
                print('*** Early stop after '+str(earlystop)+' epochs with no improvement')
                break
        comunication_data["turno"] = name
        comunication_data["autor"] = name
        comunication_data["info"]["receivings"] = sendings
        comunication_data["info"]["sendings"] = receivings


        save_json_file(comunication_data,"status_swarn.json")
        wormhole_send_model('swarn_json_model.json')
        print("Send command executed\n\n\n")
        time.sleep(40)
        comunication_data["invalid"] = True     
        comunication_data["turno"] = next_server  
        comunication_data["info"]["receivings"] = sendings
        comunication_data["info"]["sendings"] = receivings
        save_json_file(comunication_data,"status_swarn.json")
        # send_http_status(ips[next_server],"swarn")
        #send_scp_file('swarn_best_model_'+name+'.pt','swarn_best_model.pt',ips[next_server])
        #send_scp_status(ips[next_server],"swarn")
        print("SETTING STATUS A: ",comunication_data)
        print("ending")


    print(rs)
    # torch.save(model.state_dict(),'swarn_last_model_'+name+'.pt')
    print("Best model saved best_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc_max,epoch_max))
    print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    return history

def wait_all_to_finish():
    comunication_data = load_json_file("status_swarn.json")
    turno = comunication_data["turno"]
    autor = comunication_data["autor"]
    invalid = comunication_data["invalid"]
    while turno != autor or invalid:
        time.sleep(8)
        comunication_data = load_json_file("status_swarn.json")
        turno = comunication_data["turno"]
        autor = comunication_data["autor"]
        invalid = comunication_data["invalid"]
        print("esperando: ",comunication_data)

def fit_swarn_ngrok(epochs, lr,model, train_loader,val_loader, name,next_server,before_server,ips, opt_func = torch.optim.SGD, earlystop=5,rounds = 5):
    acc_max = 0
    random.seed(0)
    worm_code = random.randint(0,16777216)
    history = []
    optimizer = opt_func(model.parameters(),lr)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()
    comunication_data = set_SWARN_http_initial_status(name,0)
    c = 0
    warnings.showwarning = filter_warning
    for round in range(rounds):
        comunication_data = load_json_file("status_swarn.json")
        turno = comunication_data["turno"]
        autor = comunication_data["autor"]
        invalid = comunication_data["invalid"]
        print("r:", round,comunication_data)
        while turno != autor or (invalid and round != 0):
            c += 1
            time.sleep(8)
            comunication_data = load_json_file("status_swarn_before.json")
            turno = comunication_data["turno"]
            autor = comunication_data["autor"]
            invalid = comunication_data["invalid"]
            get_http_file(ips[next_server],"status_swarn.json","status_swarn_before.json")
            comunication_data["autor"] = name
            if round == 0 and name == "apple":
                comunication_data["invalid"] = False  
                invalid = False  
                save_json_file(comunication_data,"status_swarn.json")
            print("esperando: ",c,comunication_data,turno != autor or (invalid and round != 0))
        if round == 0 and name == "apple":
            comunication_data["invalid"] = False

        else:
            ready = comunication_data["info"]["ready"]
            print(ready)
            if ready == name:
                get_http_file(ips[before_server],"swarn_best_model.pth","swarn_best_model.pth")

        model.load_state_dict(load_json_model('swarn_json_model.json'))

        for epoch in range(epochs):
            t1 = time.time()    
            model.train()
            train_losses = []
            for batch in train_loader:
                warnings.showwarning = filter_warning
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
                st = '   ***    '

            # torch.save(model.state_dict(),'swarn_best_model.pt')
            save_json_model(model,"swarn_json_model.json")
            t2 =  "{:6.1f} ".format(time.time() - t1)
            model.epoch_end(epoch, result,st+t2)
            history.append(result)
            if epoch - epoch_max >= earlystop:
                print(rs)
                print('*** Early stop after '+str(earlystop)+' epochs with no improvement')
                break
        comunication_data["turno"] = name
        comunication_data["autor"] = name
        comunication_data["info"]["ready"] = next_server

        save_json_file(comunication_data,"status_swarn.json")
        # send_model_wormhole('swarn_best_model.pt',worm_code)
        time.sleep(80)
        comunication_data["invalid"] = True     
        comunication_data["turno"] = next_server  
        comunication_data['info']['ready'] = ''
        save_json_file(comunication_data,"status_swarn.json")
        # send_http_status(ips[next_server],"swarn")
        #send_scp_file('swarn_best_model_'+name+'.pt','swarn_best_model.pt',ips[next_server])
        #send_scp_status(ips[next_server],"swarn")
        print("SETTING STATUS A: ",comunication_data)
        print("ending")


    print(rs)
    torch.save(model.state_dict(),'swarn_last_model_'+name+'.pt')
    print("Best model saved best_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc_max,epoch_max))
    print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    return history


def fit_swarn_sftp(epochs, lr,model, train_loader,val_loader, name,first,ips, opt_func = torch.optim.SGD, earlystop=5,rounds = 5):
    acc_max = 0
    random.seed(0)
    worm_code = random.randint(0,16777216)
    history = []

    optimizer = opt_func(model.parameters(),lr)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()
    comunication_data = set_SWARN_http_initial_status(name,first)
    next_server = name + 1 if name < len(ips.keys()) else 0
    c = 0
    warnings.showwarning = filter_warning
    for round in range(rounds):
        comunication_data = load_json_file("status_swarn.json")
        turno = comunication_data["turno"]
        autor = comunication_data["autor"]
        invalid = comunication_data["invalid"]
        print("r:", round,comunication_data)
        while turno != autor or (invalid and round != 0):
            c += 1
            time.sleep(8)
            comunication_data = load_json_file("status_swarn_before.json")
            turno = comunication_data["turno"]
            autor = comunication_data["autor"]
            invalid = comunication_data["invalid"]
            comunication_data["autor"] = name
            if round == 0 and name == "kintun":
                comunication_data["invalid"] = False  
                invalid = False  
                save_json_file(comunication_data,"status_swarn.json")
            print("esperando: ",c,comunication_data,turno != autor or (invalid and round != 0))
        if round == 0 and name == "kintun":
            comunication_data["invalid"] = False

        else:
            ready = comunication_data["info"]["ready"]
            print(ready)
            
        model.load_state_dict(load_json_model('swarn_json_model.json'))

        for epoch in range(epochs):
            t1 = time.time()    
            model.train()
            train_losses = []
            for batch in train_loader:
                warnings.showwarning = filter_warning
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
                st = '   ***    '

            # torch.save(model.state_dict(),'swarn_best_model.pt')
            save_json_model(model,"swarn_json_model.json")
            t2 =  "{:6.1f} ".format(time.time() - t1)
            model.epoch_end(epoch, result,st+t2)
            history.append(result)
            if epoch - epoch_max >= earlystop:
                print(rs)
                print('*** Early stop after '+str(earlystop)+' epochs with no improvement')
                break
        comunication_data["turno"] = name
        comunication_data["autor"] = name
        comunication_data["info"]["ready"] = next_server
        
        save_json_file(comunication_data,"status_swarn.json")
        send_sftp_file(ips[next_server],"swarn_json_model.json","swarn_json_model.json")
        send_sftp_file(ips[next_server],"status_swarn.json","status_swarn_before.json")
        #send_model_wormhole('swarn_best_model.pt',worm_code)
        comunication_data["invalid"] = True     
        comunication_data["turno"] = next_server  
        comunication_data['info']['ready'] = ''
        save_json_file(comunication_data,"status_swarn.json")
        # send_http_status(ips[next_server],"swarn")
        #send_scp_file('swarn_best_model_'+name+'.pt','swarn_best_model.pt',ips[next_server])
        #send_scp_status(ips[next_server],"swarn")
        print("SETTING STATUS A: ",comunication_data)
        print("sleeping")
        time.sleep(80)
        


    print(rs)
    torch.save(model.state_dict(),'swarn_last_model_'+name+'.pt')
    print("Best model saved best_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc_max,epoch_max))
    print("Last model saved last_model.pt (val_acc = {:.4f} in epoch = {:3d})".format(acc,epoch))
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    return history



def fit_swarn_multimodel_sftp(rounds,epochs, optimizer,model, train_loader,val_loader, name,first,ips,working_dir,opt_lr,next_working_dir,model_type,data_name,writer,loss_fn, earlystop=5):
    """
    Trains a SWARN multimodel using SFTP communication protocol.

    Args:
        rounds (int): The number of training rounds.
        epochs (int): The number of epochs per round.
        optimizer: The optimizer used for training.
        model: The model to be trained.
        train_loader: The data loader for the training dataset.
        val_loader: The data loader for the validation dataset.
        name: The name of the current server.
        first: The name of the first server in the communication chain.
        ips: A dictionary containing the IP addresses of all servers.
        working_dir: The working directory for saving model files and status information.
        opt_lr: The learning rate for the optimizer.
        next_working_dir: The working directory of the next server in the communication chain.
        model_type: The type of the model.
        data_name: The name of the dataset.
        writer: The SummaryWriter object for logging training information.
        loss_fn: The loss function used for training.
        earlystop (int, optional): The number of rounds to wait for improvement in validation accuracy before early stopping. Defaults to 5.

    Returns:
        history: A list containing the training history for each round.
        acc_max: The maximum validation accuracy achieved during training.
        ep_max: The epoch at which the maximum validation accuracy was achieved.
    """
    acc_max = 0
    history = []
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    t0 = time.time()
    comunication_data = set_SWARN_http_initial_status(name,first,working_dir)
    c = 0
    print(name,name.__class__)
    if name.__class__ == int:
        next_server = name + 1 if name + 1 < len(list(ips.keys())) else 0
    else:
        index = list(ips.keys()).index(name)
        keys = list(ips.keys())
        next_key = keys[(index + 1) % len(keys)]
        next_server = ips[next_key]

        next_server = list(ips.keys())[(list(ips.keys()).index(name) + 1) % len(ips.keys())]
    
    warnings.showwarning = filter_warning
    acc_max = 0
    for round in range(rounds):
        timestamp = time.strftime("%Y-%m-%d-%H")
        comunication_data = load_json_file("status_swarn.json",working_dir)
        turno = comunication_data["turno"]
        autor = comunication_data["autor"]
        invalid = comunication_data["invalid"]
        print("r:", round,comunication_data)
        while turno != autor or invalid:
            c += 1
            time.sleep(50)
            comunication_data = load_json_file("status_swarn.json",working_dir)
            turno = comunication_data["turno"]
            autor = comunication_data["autor"]
            invalid = comunication_data["invalid"]
            comunication_data["autor"] = name
            print(list(ips.keys()),name)
            if round == 0 and name == first:
                comunication_data["invalid"] = False  
                invalid = False  
                save_json_file(comunication_data,"status_swarn.json",working_dir)
            print(round,"esperando: ",c,comunication_data,turno != autor or (invalid and round != 0))

        print('f',first,'yo',name)
            
        model.load_state_dict(torch.load(working_dir+'swarn_best_model.pth'))
        
        results, acc_max, ep_max= train_pretrained_model_summary(round=round,epochs=epochs,model= model,train_loader= train_loader,val_loader= val_loader,name=model_type,path=working_dir,acc_max=acc_max,type='swarn',rounds=rounds,writer=writer,loss_fn=loss_fn,opt_func=optimizer,earlystop=earlystop)
        history.append(results)
        comunication_data["turno"] = name
        comunication_data["autor"] = name
        comunication_data["info"]["ready"] = next_server
        save_json_file(comunication_data,"status_swarn.json",working_dir)
        send_sftp_file(ips[next_server],working_dir+"swarn_best_model.pth",next_working_dir+ "swarn_best_model.pth")
        send_sftp_file(ips[next_server],working_dir+"status_swarn.json",next_working_dir+"status_swarn.json")
        #send_model_wormhole('swarn_best_model.pt',worm_code)
        comunication_data["invalid"] = True     
        comunication_data["turno"] = next_server  
        comunication_data['info']['ready'] = ''
        save_json_file(comunication_data,"status_swarn.json",working_dir)
        # send_http_status(ips[next_server],"swarn")
        #send_scp_file('swarn_best_model_'+name+'.pt','swarn_best_model.pt',ips[next_server])
        #send_scp_status(ips[next_server],"swarn")
        print("SETTING STATUS A: ",comunication_data)
        print("sleeping")

            # Add accuracy results to SummaryWriter
        writer.add_scalars(main_tag="Round_Accuracy", 
                        tag_scalar_dict={"round_acc": acc_max}, 
                        global_step=round)
        writer.close() 
        # Close the writer
    writer_path = f'{working_dir}/results/{data_name}/swarn/{timestamp}/{len(list(ips.keys()))}_devices_{rounds}_rounds_{epochs}_epochs/'
    save_pretrained_model(model,writer_path,f'{len(list(ips.keys()))}_num_devices_{rounds}_rounds_{epochs}_epochs_{opt_lr}_optimizer.pth')

    time.sleep(60)


    print(rs)
    print(f"Best model saved swar_best_model.pt (val_acc = {acc_max} ")
    print(f"Last model saved swarn_last_model.pt (val_acc = acc in {ep_max} ")
    print(rs)
    print("Training Time: {:.2f} sec".format(time.time()-t0))


    return history, acc_max, ep_max