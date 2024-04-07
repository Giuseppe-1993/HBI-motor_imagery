from setuptools import setup, find_packages

setup(
    name="HBI_motor_imagery", # Replace with your own username
    version="0.0.1",
    author="Giuseppe Lai",
    url="git clone https://github.com/Giuseppe-1993/HBI-motor_imagery.git",
    author_email="giuseppelai93@gmail.com",
    description="Code for publication titled: Cardiac Cycle Modulates Alpha and Beta Suppression during Motor Imagery",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=2.7"
)