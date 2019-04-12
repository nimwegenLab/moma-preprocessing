from setuptools import setup

setup(name='mmpreprocesspy',
      version='0.1',
      description='Analyzing MoMA data',
      url='https://github.com/michaelmell/mmpreprocesspy',
      author='Michael Mell, Guillaume Witzi',
      author_email='michael.mell@unibas.ch',
      license='MIT',
      packages=['mmpreprocesspy'],
      zip_safe=False,
      install_requires=['numpy','scikit-image','scipy','keras','jupyter','jupyterlab','pandas','h5py','tifffile', 'trackpy','tzlocal', 'ipympl','tensorflow'],
      )
