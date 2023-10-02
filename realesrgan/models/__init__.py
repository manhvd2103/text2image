import importlib
import os
from basicsr.utils import scandir

# Auto scan and import model modules for registry
# Scan all the files that end with "_model.py" under the model folder
model_folder = os.path.dirname(os.path.abspath(__file__))
model_filenames = [os.path.splitext(os.path.basename(v))[0] for v in scandir(model_folder) if v.endswith("_model.py")]
# Import all the model modules
_model_modules = [importlib.import_module(f"realesrgan.models.{filename}") for filename in model_filenames]
