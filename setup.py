import os
from setuptools import setup, find_packages
cwd = os.path.dirname(os.path.abspath(__file__))

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(
    name='hanasu',
    version='2.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
    package_data={'': ['*.txt', 'cmudict_*']},
    entry_points={
        "console_scripts": [
            "hanasu = hanasu.main:main",
        ],
    },
)
