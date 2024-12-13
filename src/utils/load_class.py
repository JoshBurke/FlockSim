import importlib

def load_class(class_path: str):
    """Dynamically load a class from a string path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)