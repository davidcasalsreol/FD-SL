from enum import Enum
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchinfo import summary
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Dict, List
from pathlib import Path
import torch.optim as optim
from timeit import default_timer as timer 
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from utils.writer_utils import create_writer

# Define the enumeration for the model architectures
from enum import Enum

class ModelName(Enum):
    """
    Enum class representing different model names.
    
    """

    ALEXNET = "AlexNet"
    VGG16 = "VGG16"
    VGG19 = "VGG19"
    RESNET = "ResNet"
    RESNET34 = "ResNet34"
    RESNET50 = "ResNet50"
    SQUEEZENET = "SqueezeNet"
    DENSENET = "DenseNet"
    INCEPTION = "Inception v3"
    GOOGLENET = "GoogLeNet"
    SHUFFLENET = "ShuffleNet v2"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3 = "mobilenet_v3"
    RESNEXT = "ResNeXt"
    WIDERESNET50 = "Wide ResNet 50"
    WIDERESNET101 = "Wide ResNet 101"  
    MNASNET = "MNASNet"
    EFFICIENTNET_B0 = "EfficientNet_B0"
    EFFICIENTNET_B1 = "EfficientNet_B1"
    EFFICIENTNET_B2 = "EfficientNet_B2"
    EFFICIENTNET_LIGHT = "EfficientNet_Light"
    EFFICIENTNET_V2_S = "EfficientNet_V2_S"
    EFFICIENTNET_V2_M = "EfficientNet_V2_M"
    EFFICIENTNET_V2_L = "EfficientNet_V2_L"

    # CNN_Classification = "CNN_Classification"





def load_image(model,image_path):
    image = Image.open(image_path)
    transform = model.weights.transforms()
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension
    return image

def print_confusion(dt,ds,show_heatmap=0,Cnorm=1):
    
    # dt: GT, ds: Prediction
    C   = confusion_matrix(dt,ds) 
    print('Confusion Matrix:')
    print(C)
    acc = accuracy_score(dt,ds) 
    acc_st = "{:.2f}".format(acc*100)
    print('Accuracy = '+str(acc_st))
    if show_heatmap:
      sns.heatmap(C/Cnorm, annot=True, cbar=None, cmap="Blues")
      plt.title("Confusion Matrix: Acc ="+acc_st)
      plt.tight_layout()
      plt.ylabel("True Class"), plt.xlabel("Predicted Class")
      plt.show()

def prediction_step(model, batch):
    
    images, labels = batch 
    out = model(images)  # Generate predictions
    return out

def prediction(model, val_loader):
    
    model.eval()
    pred = [prediction_step(model, batch) for batch in val_loader]
    return pred

def get_prediction(model,dataset):
  
  set_dl = DataLoader(dataset, len(dataset), shuffle = False, num_workers = 2, pin_memory = True)
  y      = prediction(model,set_dl)
  y1     = torch.cat(y)
  y2     = y1.detach().numpy()
  ypred  = np.argmax(y2,axis = 1)
  return ypred        

def get_labels(model,dataset):
  set_dl = DataLoader(dataset, len(dataset), shuffle = False, num_workers = 2, pin_memory = True)
  for batch in set_dl:
    img,label = batch
  ygt = label.numpy()
  return ygt  


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(model, batch):
    images, labels = batch
    out = model(images)  # Forward pass
    loss = F.cross_entropy(out, labels)  # Compute loss
    acc = accuracy(out, labels)  # Compute accuracy
    return {'val_loss': loss, 'val_acc': acc}

def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end(epoch, result, st=""):
    x = np.zeros((3,))
    x[0] = result['train_loss']
    x[1] = result['val_loss']
    x[2] = result['val_acc']
    print("%5d %11.4f %11.4f %11.4f %s" % (epoch, x[0], x[1], x[2], st))



# Function to load the model with pretrained weights
def load_pretrained_model(model_name_str, weights=False, light=False):
    """
    Loads a pretrained model based on the given model name.

    Args:
        model_name_str (str): The name of the model to load.
        weights (bool, optional): Whether to return the weights of the model. Defaults to False.

    Returns:
        torch.nn.Module or str: The loaded model or the weights of the model if `weights` is True.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if light:
        return load_pretrained_model_light(model_name_str, weights)
    print(model_name_str)
    model_name = ModelName(model_name_str)

    if model_name == ModelName.ALEXNET:
        w = torchvision.models.AlexNet_Weights.DEFAULT
        model = torchvision.models.alexnet(weights=w)
    elif model_name == ModelName.VGG16:
        w = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=w)
    
    elif model_name == ModelName.VGG19:
        w = models.VGG19_Weights.DEFAULT
        model = models.vgg19(weights=w)

    elif model_name == ModelName.RESNET:
        w = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=w)
        
    elif model_name == ModelName.RESNET34:
        w = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=w) 
    elif model_name == ModelName.RESNET50:
        w = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=w) 

    elif model_name == ModelName.SQUEEZENET:
        w = models.SqueezeNet1_0_Weights.DEFAULT
        model = models.squeezenet1_0(weights=w)

    elif model_name == ModelName.DENSENET:
        w = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=w) 

    elif model_name == ModelName.INCEPTION:
        w = models.Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=w)

    elif model_name == ModelName.GOOGLENET:
        w = models.GoogLeNet_Weights.DEFAULT
        model = models.googlenet(weights=w)
        
    elif model_name == ModelName.SHUFFLENET:
        w = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        model = models.shufflenet_v2_x1_0(weights=w)
    
    elif model_name == ModelName.RESNEXT:
        w = models.ResNeXt50_32X4D_Weights.DEFAULT
        model = models.resnext50_32x4d(weights=w)
        #
    elif model_name == ModelName.WIDERESNET50:
        w = models.Wide_ResNet50_2_Weights.DEFAULT
        model = models.wide_resnet50_2(weights=w)

    elif model_name == ModelName.WIDERESNET101:
        w = models.Wide_ResNet101_2_Weights.DEFAULT
        model = models.wide_resnet101_2(weights=w)

    elif model_name == ModelName.MNASNET:
        w = models.MNASNet1_0_Weights.DEFAULT
        model = models.mnasnet1_0(weights=w)
    
    elif model_name == ModelName.EFFICIENTNET_B0:
        w = models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=w)

    elif model_name == ModelName.EFFICIENTNET_B1:
        w = models.EfficientNet_B1_Weights.DEFAULT
        model = torchvision.models.efficientnet_b1(weights=w)

    elif model_name == ModelName.EFFICIENTNET_B2:
        w = models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=w)

    elif model_name == ModelName.EFFICIENTNET_V2_L:
        w = models.EfficientNet_V2_L_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_l(weights=w)

    elif model_name == ModelName.EFFICIENTNET_V2_M:
        w = models.EfficientNet_V2_M_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_m(weights=w)
        
    elif model_name == ModelName.EFFICIENTNET_V2_S:
        w = models.EfficientNet_V2_S_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_s(weights=w)
        
    else:
        raise ValueError(f"Model {model_name_str} not recognized.")
    
    if weights:
        return w
    
    return model



# Function to load the model with pretrained weights
def load_pretrained_model_light(model_name_str, weights=False):
    """
    Loads a pretrained model based on the given model name.

    Args:
        model_name_str (str): The name of the model to load.
        weights (bool, optional): Whether to return the weights of the model. Defaults to False.

    Returns:
        torch.nn.Module or str: The loaded model or the weights of the model if `weights` is True.

    Raises:
        ValueError: If the model name is not recognized.
    """
    print(model_name_str)
    model_name = ModelName(model_name_str)

    if model_name == ModelName.ALEXNET:
        w = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        model = torchvision.models.alexnet(weights=w)
    elif model_name == ModelName.VGG16:
        w = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=w)
    
    elif model_name == ModelName.VGG19:
        w = models.VGG19_Weights.IMAGENET1K_V1
        model = models.vgg19(weights=w)

    elif model_name == ModelName.RESNET:
        w = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=w)
        
    elif model_name == ModelName.RESNET34:
        w = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=w) 
    elif model_name == ModelName.RESNET50:
        w = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=w) 

    elif model_name == ModelName.SQUEEZENET:
        w = models.SqueezeNet1_0_Weights.IMAGENET1K_V1
        model = models.squeezenet1_0(weights=w)

    elif model_name == ModelName.DENSENET:
        w = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=w) 

    elif model_name == ModelName.INCEPTION:
        w = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=w)

    elif model_name == ModelName.GOOGLENET:
        w = models.GoogLeNet_Weights.IMAGENET1K_V1
        model = models.googlenet(weights=w)
        
    elif model_name == ModelName.SHUFFLENET:
        w = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        model = models.shufflenet_v2_x1_0(weights=w)
    
    elif model_name == ModelName.RESNEXT:
        w = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        model = models.resnext50_32x4d(weights=w)
        #
    elif model_name == ModelName.WIDERESNET50:
        w = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        model = models.wide_resnet50_2(weights=w)

    elif model_name == ModelName.WIDERESNET101:
        w = models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
        model = models.wide_resnet101_2(weights=w)

    elif model_name == ModelName.MNASNET:
        w = models.MNASNet1_0_Weights.IMAGENET1K_V1
        model = models.mnasnet1_0(weights=w)
    
    elif model_name == ModelName.EFFICIENTNET_B0:
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=w)

    elif model_name == ModelName.EFFICIENTNET_B1:
        w = models.EfficientNet_B1_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b1(weights=w)

    elif model_name == ModelName.EFFICIENTNET_B2:
        w = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b2(weights=w)

    elif model_name == ModelName.EFFICIENTNET_V2_L:
        w = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_v2_l(weights=w)

    elif model_name == ModelName.EFFICIENTNET_V2_M:
        w = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_v2_m(weights=w)
        
    elif model_name == ModelName.EFFICIENTNET_V2_S:
        w = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_v2_s(weights=w)
        
    else:
        raise ValueError(f"Model {model_name_str} not recognized.")
    
    if weights:
        return w
    
    return model

def personalize_pretrained_model_2(model, class_names):
    """
    Personalizes a pretrained model by freezing its parameters, recreating the classifier layer, and setting the manual seeds.
    Modify the seeds to randomize more between models, beware that it may change convergence speed.
    Args:
        model (torch.nn.Module): The pretrained model to be personalized.
        class_names (list): A list of class names for the classification task.

    Returns:
        torch.nn.Module: The personalized model with the updated classifier layer.
    """
    for param in model.parameters():
        param.requires_grad = False
    print(len(list(model.parameters())))

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,  # same number of output units as our number of classes
                        bias=True))

    return model



def personalize_pretrained_model(model, class_names):
    """
    Personalizes a pretrained model by replacing the last layer of the classifier with a new layer
    that matches the number of output classes.
    Add new model architectures to the function as needed.
    Args:
        model (nn.Module): The pretrained model to be personalized.
        class_names (list): A list of class names.

    Returns:
        nn.Module: The personalized model.

    Raises:
        NotImplementedError: If the model architecture is not supported for customization.
    """
    # Congelar todos los parámetros del modelo
    for param in model.parameters():
        param.requires_grad = False

    print(f"Total parameters in the model: {len(list(model.parameters()))}")

    # Establecer las semillas manuales para reproducibilidad
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Obtener el número de clases
    output_shape = len(class_names)
    # Reemplazar la última capa del clasificador según la arquitectura del modelo
    if isinstance(model, models.ResNet):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, output_shape)
        for param in model.fc.parameters():
            param.requires_grad = True
    elif isinstance(model, models.VGG) or isinstance(model, models.AlexNet):
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, output_shape)
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
    elif isinstance(model, models.GoogLeNet):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, output_shape)
        for param in model.fc.parameters():
            param.requires_grad = True
    elif isinstance(model, models.Inception3):
        # Replace the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, output_shape)
        # Modify the auxiliary classifier if present
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, output_shape)
        for param in model.fc.parameters():
            param.requires_grad = True
        
    elif isinstance(model, models.DenseNet):
        # Replace the classifier for DenseNet
        model.classifier = nn.Linear(model.classifier.in_features, output_shape)
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif isinstance(model, models.ShuffleNetV2):
        # Replace the classifier for ShuffleNet v2
        model.fc = nn.Linear(model.fc.in_features, output_shape)
        for param in model.fc.parameters():
            param.requires_grad = True
    elif isinstance(model, models.MNASNet):
        # Replace the classifier for MNASNet
        if isinstance(model.classifier, nn.Sequential):
        # Replace the last layer in the Sequential container
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, output_shape)
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'features') and hasattr(model, 'classifier'):
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, output_shape)
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
    else:
        raise NotImplementedError("Model architecture not supported for customization")

    return model

def predict_personalize_model(model, image_path, class_names):
    """
    Predicts the class of an image using a personalized model.

    Args:
        model (nn.Module): The personalized model.
        image_path (str): The path to the image to predict.
        class_names (list): A list of class names.
    """
    # Load the image
    image = load_image(model,image_path)
    # Make a prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image)
    

    result_dict = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    # Get the class name]
    # Display the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # Display the prediction
    print(f"Prediction: {result_dict}")

def train_pretrained_model(epochs, model, train_loader, val_loader, name, path, optimizer, loss_fn, acc_max, writer, earlystop=10):
    """
    Trains a pretrained model using the specified parameters.

    Args:
        epochs (int): The number of training epochs.
        model: The pretrained model to be trained.
        train_loader: The data loader for the training dataset.
        val_loader: The data loader for the validation dataset.
        name (str): The name of the model.
        path (str): The path to save the trained model.
        optimizer: The optimizer used for training.
        loss_fn: The loss function used for training.
        acc_max: The maximum accuracy threshold.
        writer: The writer object for logging training metrics.
        earlystop (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. IMAGENET1K_V1s to 10.

    Returns:
        results: The training results.

    """
    history = []
    print(optimizer)
    rs = '------------------------------------------------------------------'
    #               xxxxxxxxxx  xxxxxxxxxx  xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_time = timer()
    
    # Setup training and save the results
    results = train(model=model,
                    train_dataloader=train_loader,
                    test_dataloader=val_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=epochs,
                    name=name,
                    path=path,
                    acc_max=acc_max,
                    writer=writer)
        

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    return results


def train_pretrained_model_summary(type, round, epochs, model, train_loader, val_loader, path, loss_fn, writer, opt_func, acc_max, earlystop=10):
    """
    Trains a pretrained model and returns the training results, maximum accuracy, and epoch at which maximum accuracy was achieved.

    Args:
        type (str): The type of training. Can be "fed" for federated learning or "swarn" for swarm learning.
        rounds (int): The number of training rounds.
        epochs (int): The number of epochs per training round.
        model: The pretrained model to be trained.
        train_loader: The data loader for the training dataset.
        val_loader: The data loader for the validation dataset.
        name (str): The name of the model file to be saved.
        path (str): The path where the model file will be saved.
        loss_fn: The loss function used for training.
        writer: The writer object for logging training metrics.
        opt_func: The optimizer function used for training.
        acc_max: The maximum accuracy achieved during training.
        earlystop (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 10.

    Returns:
        results: The training results.
        acc_max: The maximum accuracy achieved during training.
        epoch_max: The epoch at which maximum accuracy was achieved.
    """
    rs = '------------------------------------------------------------------'
    print('Epoch    Train-Loss   Val-Loss    Val-Acc   Best    Time [sec]')
    print(rs)
    print(type)
    start_time = timer()
    if type == "fed":
        file = "fed_local_model.pth"
    elif type == "swarn":
        file = "swarn_best_model.pth"
    results, acc_max, epoch_max = train(model=model,
                                        train_dataloader=train_loader,
                                        test_dataloader=val_loader,
                                        optimizer=opt_func,
                                        loss_fn=loss_fn,
                                        epochs=epochs,
                                        name=file,
                                        path=path,
                                        acc_max=acc_max,
                                        writer=writer,
                                        round = round)
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    return results, acc_max, epoch_max


def save_pretrained_model(model,target_dir,model_name):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = str(target_dir_path)+'/' + model_name
    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=str(model_save_path))



def train_step(model: torch.nn.Module, 
               dataloader: DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    print("training")
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        # X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: DataLoader, 
              loss_fn: torch.nn.Module,
              ) -> Tuple[float, float]:
    """
    Perform a single testing step on the given model using the provided dataloader and loss function.

    Args:
        model (torch.nn.Module): The model to be tested.
        dataloader (DataLoader): The dataloader containing the test data.
        loss_fn (torch.nn.Module): The loss function used to calculate the loss.

    Returns:
        Tuple[float, float]: A tuple containing the average test loss and test accuracy.
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            # X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    print(test_loss,test_acc)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          name: str,
          path: str,
          acc_max: float,
          writer :Optional[Any],
          round: int,
         ) -> Tuple[Dict[str, List],float,int]:
    # Create empty results dictionary
    """
        Trains a given model using the provided training and testing data.

        Args:
                model (torch.nn.Module): The model to be trained.
                train_dataloader (DataLoader): The data loader for the training data.
                test_dataloader (DataLoader): The data loader for the testing data.
                optimizer (torch.optim.Optimizer): The optimizer used for training.
                loss_fn (torch.nn.Module): The loss function used for training.
                epochs (int): The number of epochs to train the model.
                name (str): The name of the model.
                path (str): The path to save the trained model.
                acc_max (float): The maximum accuracy achieved during training.
                writer (Optional[Any]): Optional SummaryWriter for logging training progress.

        Returns:
                Tuple[Dict[str, List], float, int]: A tuple containing the results dictionary, 
                the maximum accuracy achieved, and the epoch at which the maximum accuracy was achieved.
    """
    epoch_max = 0
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }    
    # Make sure model on target device
    # model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(epoch)
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          )
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          )

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {test_loss:.4f} | "
          f"val_acc: {test_acc:.4f}"
        )
        print(test_acc,acc_max)
        if test_acc>acc_max:
                epoch_max = epoch
                acc_max = test_acc
                st = '   ***    '
                print('MEJORAaaa')
                print(path,name)
                save_pretrained_model(model,path,name)

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if writer:
        # Add loss results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                            tag_scalar_dict={"train_loss": train_loss,
                                                "val_loss": test_loss},
                            global_step=round*epochs+epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag="Accuracy", 
                            tag_scalar_dict={"train_acc": train_acc,
                                                "val_acc": test_acc}, 
                            global_step=epoch)
            
            # Close the writer
            writer.close()
        else: 
            pass


    # Return the filled results at the end of the epochs
    return results, acc_max, epoch_max
