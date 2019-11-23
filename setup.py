from setuptools import setup
setup(
    name='fahr',
    packages=['fahr'],
    install_requires=['docker', 'jinja2', 'click'],
    extras_require={
        # drivers require addtl packages
        'sagemaker': ['boto3', 'sagemaker'],
        'kaggle': ['kaggle'],
        # all installs all addtl packages
        'all': ['boto3', 'sagemaker', 'kaggle'],
        # develop is all addtl packages plus dev stuff
        'develop': [
            'pytest', 'pytest-socket',  # testing
            'sphinx', 'sphinx_rtd_theme',  # docs
            'boto3', 'sagemaker',  # sagemaker
            'kaggle'  # kaggle
        ]
    },
    py_modules=['fahr'],
    version='0.0.1',
    description='Tool for running remote machine learning jobs remotely.',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/fahr',
    download_url='https://github.com/ResidentMario/fahr/tarball/0.0.1',
    keywords=['data', 'machine learning', 'data engineering'],
    classifiers=[],
    include_package_data=True,
    entry_points='''
        [console_scripts]
        fahr=fahr.cli:cli
    ''',
)