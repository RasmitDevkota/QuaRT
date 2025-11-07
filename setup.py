from setuptools import setup, find_packages

setup(
    name="QuaRT",
    version="0.1.0",
    author="Computational Cosmology at Georgia Tech",
    author_email="rdevkota3@gatech.edu",
    url="https://github.com/RasmitDevkota/QuaRT/",
    license="LICENSE",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "qiskit",
        "qiskit-aer",
        "qiskit-addon-utils",
        "qiskit-experiments",
        "qiskit-ibm-runtime",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx-autoapi",
            "sphinx-rtd-theme",
            "python-markdown-math",
        ],
    },
)

