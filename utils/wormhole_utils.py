
import os
import subprocess
import time

from utils.json_utils import load_json_file

directory = "/home/dcasals/TFG/"


def wormhole_receive_model(phrase, name):
  """
  Receives a file using the wormhole protocol.

  Args:
    phrase (str): The wormhole phrase to use for receiving the file.
    name (str): The name of the file to be received.

  Raises:
    Exception: If an error occurs during the file receiving process.

  Returns:
    None
  """
  try:
    rta = 'Y'
    print(phrase)
    comand = f"echo {rta} | {phrase} -o {directory}{name}"
    # Lee el comando del archivo
    # Verifica que se haya leído un comando
    print("VOYYYY :", comand)

    p = subprocess.Popen(comand, shell=True)

    # Imprime la salida del comando wormhole receive
    if p.returncode == 0:
      print(p.stdout)
    else:
      print(f"Error receiving file: {p.stderr}")
  except Exception as e:
    print(f"An error occurred: {e}")


def get_all_models_wormhole(nodes):
  """
  Retrieves all models using the wormhole protocol for the given nodes.

  Args:
    nodes (list): A list of node names.

  Raises:
    Exception: If an error occurs during the process.

  Returns:
    None
  """
  for node in nodes: 
    try:
      status = load_json_file(node + "_status_fed.json")
      print('\n.\n\n \n0obteniendo todos los modelos\n.\n\n \n')
      
      phrase = status[node]["info"]["code"]
      name = node + "_fed_local_model.json"
      rta = 'Y'
      comand = f'echo {rta} | {phrase} -o {directory}{name}'
      
      # Lee el comando del archivo
      # Verifica que se haya leído un comando
      print("VOYYYY :", comand)

      p = subprocess.run(comand, shell=True)
      
      # Imprime la salida del comando wormhole receive
      if p.returncode == 0:
        print(p.stdout)
      else:
        print(f"Error receiving file: {p.stderr}")
    except Exception as e:
      print(f"An error occurred: {e}")
      raise e



def get_last_non_empty_line(file_path):
  """
  Retrieves the last non-empty line from a given file.

  Args:
    file_path (str): The path to the file.

  Returns:
    str: The last non-empty line in the file, or an empty string if the file is empty or does not exist.
  """
  with open(directory + file_path, 'r') as file:
    lines = file.readlines()
  
  # Find the last non-empty line
  for line in reversed(lines):
    if line.strip():
      return line.strip()
  return ''



def wormhole_send_model(name):
  """
  Sends a model using the Wormhole protocol.

  Args:
    name (str): The name of the model to send.

  Returns:
    str: The last non-empty line from the 'stderr.txt' file.

  Raises:
    FileNotFoundError: If the 'stdout.txt' or 'stderr.txt' file is not found.

  """
  # Open files for stdout and stderr
  try:
    stdout_file = open(directory + "stdout.txt", "w")
    stderr_file = open(directory + "stderr.txt", "w")
  except FileNotFoundError:
    stdout_file = open(directory + "stdout.txt", "x")
    stderr_file = open(directory + "stderr.txt", "x")

  # Fork the process
  pid = os.fork()

  if pid == 0:  # Child process
    # Redirect stdout and stderr to files
    os.dup2(stdout_file.fileno(), 1)
    os.dup2(stderr_file.fileno(), 2)

    # Execute the command in the child process
    subprocess.run([f"wormhole send {directory}{name}"], shell=True, check=True)
    os._exit(0)

  else:  # Parent process
    # Close file descriptors in the parent process
    time.sleep(5)
    last_line = get_last_non_empty_line('stderr.txt')
    print("LLAAAAAAST:", last_line)
    return last_line
        
