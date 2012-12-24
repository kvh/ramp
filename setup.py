from setuptools import setup, find_packages
import os

f = open(os.path.join(os.path.dirname(__file__), 'README.md'))
long_description = f.read()
f.close()

version = '0.1.4'

setup(
    name='ramp',
    version=version,
    description="Rapid machine learning prototyping",
    long_description=long_description,
    classifiers=[
        'License :: OSI Approved :: BSD License'
    ],
    keywords='machine learning data analysis statistics mining',
    author='Ken Van Haren',
    author_email='kvh@science.io',
    url='http://github.com/kvh/ramp',
    license='BSD',
    packages=find_packages(exclude=["*.tests"]),
    zip_safe=False
)

