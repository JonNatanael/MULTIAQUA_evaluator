import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multiaqua_evaluator",
    version="0.1",
    author="Jon Muhovič",
    author_email="jon.muhovic@fri.uni-lj.si",
    description="Evaluation toolkit for the MULTIAQUA dataset.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonNatanael/MULTIAQUA_evaluator",
    packages=['multiaqua_eval'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'Pillow',
        'numpy',
        'yacs',
        'tqdm',
        'opencv-python',
        'pandas'
    ]
)
