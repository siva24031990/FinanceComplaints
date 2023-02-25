"""Basic imports for the steup file"""
from setuptools import find_packages, setup
from typing import List

#Details of the Package
PROJECT_NAME = 'mlpractice'
VERSION = "0.1.0"
AUTHOR = "SIVA ELUMALAI"
DESCRIPTION = "project to know how the structure of a ML project"
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements_list() -> List[str]:
    """
    to get the requiement details of this project to be installed
    """
    with open(REQUIREMENT_FILE_NAME, "r") as requirements_file:
        requirements_list = requirements_file.readlines()
        requirements_list = [requirement_name.replace("\n","") for requirement_name in requirements_list]
        if HYPHEN_E_DOT in requirements_list:
            requirements_list.remove(HYPHEN_E_DOT)
    return requirements_list

setup(
    name = PROJECT_NAME,
    author = AUTHOR, 
    description = DESCRIPTION, 
    version = VERSION, 
    packages = find_packages(), 
    install_requires = get_requirements_list()
)

        


