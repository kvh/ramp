from setuptools import setup, find_packages

version = '0.1'

setup(name='ramp',
      version=version,
      description="Rapid machine learning prototyping",
      long_description=open("README.md").read(),
      classifiers=[
          'License :: OSI Approved :: BSD License'
      ],
      keywords='machine learning data analysis statistics mining',
      author='Ken Van Haren',
      author_email='kvh@science.io',
      url='http://github.com/kvh/ramp',
      license='BSD',
      packages=find_packages('ramp'),
      package_dir={'': 'ramp'},
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'numpy',
          'pandas',
      ]
      )

