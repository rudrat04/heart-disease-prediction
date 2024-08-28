import sys
import os

# Add the path to your app
path = 'D:\Projects\Heart_Disease_Prediction'
if path not in sys.path:
    sys.path.append(path)

# Set the Flask app
from app import app as application

# Activate the virtual environment if you have one
activate_this = '.venv/Scripts/activate'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))
