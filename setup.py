from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    """
    This function reads the requirements file and returns a list of requirements.

    Args:
        file_path (str): The path to the requirements.txt file.

    Returns:
        List[str]: A list of package requirements excluding '-e .'
    """

    requirements = []
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.replace("\n"," ") for req in requirements]

            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

    return requirements


# Package setup configuration
setup(
    name = 'MLproject',
    version = '0.0.1',
    author = "Abhishek",
    author_email= "abhichris63@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)