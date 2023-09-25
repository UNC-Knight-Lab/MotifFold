from setuptools import setup, find_packages

VERSION = '1.0.1'
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# Setting up
setup(
    name="seq_match",
    version=VERSION,
    author="Supraja Chittari",
    author_email="suprajac@email.unc.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/UNC-Knight-Lab/MotifFold',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'sklearn',
        'seaborn',
        'category_encoders'
    ],
    entry_points={
        'console_scripts': ['motif-fold=motif-fold.command_line:main'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
    ],
    license='GPL-3.0'
)