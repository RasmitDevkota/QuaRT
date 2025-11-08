# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "QuaRT-LBM"
copyright = "2025, Rasmit Devkota and John Wise"
author = "Rasmit Devkota and John Wise"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]

source_suffix = ".rst"

exclude_patterns = [
    ".ipynb",
    "**/**.ipynb_checkpoints",
    "**/*-checkpoint/",
    "*-checkpoint/",
    "**-checkpoint/",
    "**/**-checkpoint/",
    "../../QuaRT/**/.ipynb_checkpoints/"
]

# -- Options for Sphinx AutoAPI ----------------------------------------------

autoapi_dirs = [
    "../../quart-lbm/",
]

autoapi_ignore = [
    "qiskit"
    "qiskit/"
    "**/qiskit"
    "qiskit/**"
    "**/qiskit/**"
]

autodoc_inherit_docstrings = False

autoapi_python_class_content = "class"

autodoc_typehints = 'description'

import inspect
from qiskit.circuit import QuantumCircuit # We need this to check its module path

# Get the base module path for Qiskit to use for comparison
QISKIT_MODULE_PATH = inspect.getmodule(QuantumCircuit).__name__.split('.')[0]
print(QISKIT_MODULE_PATH)

def autoapi_skip_member(app, what, name, obj, skip, options):
    """
    Skip all members inherited from a class whose defining module is 'qiskit'.
    """

    try:
        print(obj.docstring)
    except:
        pass
    try:
        print(obj._docstring)
    except:
        pass
    try:
        print(obj.docstring_resolved)
    except:
        pass
    
    # 1. Check if the object is an inherited member
    if obj.inherited:
        try:
            # 2. Use introspection to find the object's origin
            # obj.qualname is the fully qualified name (e.g., LogicalCircuit.draw)
            # We get the actual Python object that defines this member.
            defining_module = inspect.getmodule(obj.obj).__name__
            
            # 3. Compare the defining module's root against 'qiskit'
            if defining_module.startswith(QISKIT_MODULE_PATH):
                # Skip the member if it originated in Qiskit
                return True
            else:
                print(defining_module)
                
        except (AttributeError, TypeError):
            # If introspection fails for some members (e.g., due to static analysis)
            # fall back to the default behavior.
            pass

    # Don't skip if it's not inherited or not from Qiskit
    return skip

def setup(app):
    app.connect('autoapi-skip-member', autoapi_skip_member)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

