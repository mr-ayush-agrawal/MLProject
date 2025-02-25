from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_req(file_path : str) -> List[str]:
    '''
    This Function returns the list of the requirments present in the file_path
    '''
    requirments = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments = [req.replace("\n", "") for req in requirments]
    
    if HYPHEN_E_DOT in requirments : 
        requirments.remove(HYPHEN_E_DOT)

    return requirments

setup(
    name = 'ML Project',
    version='0.0.1',
    author='Ayush',
    packages=find_packages(),
    install_requires = get_req('requirments.txt')
)