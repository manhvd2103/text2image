import importlib
import os
from basicsr.utils import scandir

arch_folder = os.path.dirname(os.path.abspath(__file__))
arch_filenames = [os.path.splitext(os.path.basename(v))[0] for v in scandir(arch_folder) if v.endswith("_arch.py")]
# Import all the arch modules
_arch_modules = [importlib.import_module(f"realesrgan.archs.{filename}") for filename in arch_filenames]
