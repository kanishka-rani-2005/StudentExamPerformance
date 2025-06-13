from setuptools import find_packages,setup
from typing import List


# // creating function so that setup will take libraries
# from requirements .txt and install them 
def get_requirements(file_path:str)->List[str]:
    """Returns a list of requirements from a requirements.txt file."""
    requirements=[]
    with open (file_path) as f:
        # \n will also get read by this function
        requirements=f.readlines()
        # removing the \n from the end of each line
        requirements=[r.replace("\n","") for r in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
        
    return requirements   


setup(
name='Mlproject',
version='0.0.1',
author='Kanishka',
author_email='kanishka22043@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)

