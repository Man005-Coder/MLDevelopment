from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->list[str]:
    requirement=[]
    with open(file_path) as f:
        requirement= f.readlines()
        requirement=[req.replace("\n","") for req in requirement]

        if  HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)

    return requirement

setup(
    name="First Model Package",
    version="0.0.0",
    author="Manthan Raval",
    author_email="manthanrraval@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    description="My first model package to learn new things",
    python_requires=">=3.13.14",
)
