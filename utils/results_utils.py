import json
import os
import queue
import zipfile
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple

from PIL import Image



def get_input(prompt):
    """Gets input from the user."""
    while True:
        try:
            value = input(prompt)
            return value
        except ValueError:
            print("Invalid input. Please enter a valid value.")


def show_results(image_path, outputs,labels):
    _, indices = torch.sort(outputs, descending=True)
    percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    
    # Display the image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    # Display the top 5 predictions
    print("Top 5 predictions:")
    for idx in indices[0][:5]:
        print(f'{labels[idx]}: {percentage[idx].item():.2f}%')



def get_bests_models_from_json(n, json_file, directory, train_type=None, model_name=None):
    """
    Retrieves the best models from a JSON file based on test accuracy.

    Args:
        n (int): The number of best models to retrieve.
        json_file (str): The name of the JSON file.
        directory (str): The directory where the JSON file is located.
        train_type (str, optional): The type of training. Defaults to None.
        model_name (str, optional): The name of the model. Defaults to None.

    Returns:
        list: A list of dictionaries representing the best models.
    """
    with open(directory + json_file, 'r') as f:
        data = json.load(f)

    # Create an ordered priority queue
    print(data)
    data = data['items']
    # Insert items into the priority queue
    c = 0
    print(train_type, model_name,len(data))
    lista_ordenada = sorted(data, key=lambda x: x["test_accuracy"], reverse=True)
    print(len(lista_ordenada)) 
    # Get the top n models based on test accuracy
    best_models = []
    for i in range(n):
        item = lista_ordenada[i]
        print(item)
        best_models.append(item)
    print('gone')
    return best_models


def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
):
    
    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_image_pred = model(transformed_image)



    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    print(img)
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs}"
    )
    plt.text(0, 0, f"Pred: {class_names} | Prob: {target_image_pred_probs}")
    plt.axis(False)