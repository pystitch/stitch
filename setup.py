from setuptools import setup, find_packages
from os import path

import versioneer


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='knotr',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    description='Reproducible report generation tool.',
    long_description=long_description,

    url='https://github.com/tomaugspurger/stitch',

    author='Tom Augspurger',
    author_email='tom.augspurger88@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # keywords='sample setuptools development',
    packages=find_packages(exclude=['docs', 'tests']),

    install_requires=['jupyter_core', 'traitlets', 'ipython', 'jupyter_client',
                      'nbconvert', 'pandocfilters', 'pypandoc', 'click'],

    extras_require={
        'dev': ['pytest', 'pytest-cov', 'sphinx', 'pandas', 'matplotlib'],
    },
    # package_data={
    #     'sample': ['package_data.dat'],
    # },
    entry_points={
        'console_scripts': [
            'stitch=stitch.cli:cli',
        ],
    },
)
