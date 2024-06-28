
import os
import string
import subprocess

import torch
from utils.model_utils import load_pretrained_model, personalize_pretrained_model, predict_personalize_model
from utils.results_utils import get_bests_models_from_json, get_input, show_results


def interactive_visualizer():
    """
    Interactive visualizer for displaying the best models.

    This function prompts the user to enter the number of models to receive and optional filters for train type and model name.
    It then retrieves the best models from a JSON file and displays their information.

    Returns:
        list: A list of dictionaries representing the best models.

    """
    n = int(input("Enter the number of models to receive: "))
    train_type = get_input("Enter the train type to filter by (leave blank for no filter): ") or 'None'
    model_name = get_input("Enter the model name to filter by (leave blank for no filter): ") or 'None'
    path = str(os.path.dirname(__file__))
    # Obtener la parte de la ruta hasta "TFG"
    print(path)
        
    exp  = os.listdir(path+'/experiments')
    exp = [x for x in exp if x.endswith("summary.json")]
    d = dict(zip(range(len(exp)),exp))
    for x in d.items():
        print(x)
    json_file = get_input("Enter the name of the execution_experiment JSON: or N for diferent path:") or 'N'
    if json_file == 'N':
        json_file = get_input("Enter the name of the execution_experiment JSON: ")
    json_file = d[int(json_file)]
    best_models = get_bests_models_from_json(n,json_file, path+'/experiments/' ,train_type, model_name)
    print(len(best_models))
    for i, model in enumerate(best_models, 1):
        print(f"Model {i}:")
        print(f" - Experiment: {model['experiment']}")
        print(f" - Model: {model['model']}")
        print(f" - Train Type: {model['type']}")
        print(f" - Test Accuracy: {model['test_accuracy']}")
        print(f" - Path to Model: {model['path_to_model']}")
        print(f" - Path to Grafics: {model['writer_logs']}")
        print()
    return best_models, path+'/experiments/',json_file




if __name__ == '__main__':
    test_file = get_input("Enter the name of the test file (default /Users/admin/TFG/datasets/pecas02/test): ") or '/Users/admin/TFG/datasets/pecas02/test'
    models, zip_dir, file = interactive_visualizer()
    save_zip = get_input("Do you want to save the selected models in a zip file? (y/n): ") or 'n'
    paths = []
    time = file.split("_")[0]
    if save_zip == 'y':
        for model in models:
            model_location = model['path_to_model']
            paths.append(model_location)
            log_path = model['writer_logs']
            if log_path.endswith('/'):
                log_path = log_path + '*'
            paths.append(log_path)
        working_dir = os.path.dirname(__file__)
        name = get_input(f"Enter the name of the zip file (default {time}): ") or time
        all_paths = ' '.join(paths)
        zip_coomand = f'zip -r {working_dir}/{name}.zip . -i {all_paths}'
        subprocess.run(zip_coomand, shell=True)
    delete_zip = get_input("Do you want to delete the all models zip? (y/n): ") or 'n'
    if delete_zip == 'y':
        rm_coomand = f'rm {zip_dir}{file}_execution_experiment.zip'
        subprocess.run(rm_coomand, shell=True)

    # predict = get_input("Do you want to predict with the selected models? (y/n): ") or 'n'
    # if predict == 'y':
    #     image = get_input("Enter the path of the image: ") or '/Users/admin/TFG/datasets/pecas02/test/class_4/ISIC_0028418_00.jpg'
    #     for model in models:
    #         model_location = model['path_to_model']
    #         model_name = model['model']
    #         model = load_pretrained_model(model_name)
    #         model = personalize_pretrained_model(model, model_name)
    #         model.load_state_dict(torch.load(model_location))
    #         predict_personalize_model(model, image,)
    #         print(model_location)
    #         print(model_name)
    #         print("Predicting with model...")
    #         predict(model_location)


    # m = get_input("Do you want to predict and visualize another models? (y/n): ") or 'n'
    # path = get_input("Enter the path of the models: ") or 'models/pecas01/Swarn/2024-06-26-00/AlexNet_2_rounds_3_epochs_Adam_0.001.pth'
    # while m == 'y':

    #     #predict(path)
    #     model = get_input("Do you want to predict and visualize another models? (y/n): ") or 'n'
    #     path = get_input("Enter the path of the models: ") or 'models/pecas01/Swarn/2024-06-26-00/AlexNet_2_rounds_3_epochs_Adam_0.001.pth'
    #     model_name = string.split('_')[0]
    #     model = load_pretrained_model(model_name)
    #     model = personalize_pretrained_model(model, model_name)
    #     model.load_state_dict(torch.load(path))
    #     show_results(model)

    #     m = get_input("Do you want to predict and visualize another models? (y/n): ") or 'n'



