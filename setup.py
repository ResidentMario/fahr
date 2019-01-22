from setuptools import setup
setup(
    name='alekseylearn',
    packages=['alekseylearn'], # this must be the same as the name above
    install_requires=['docker', 'jinja2', 'click'],
    extras_require={
        'sagemaker': ['boto3', 'sagemaker']
    },
    py_modules=['alekseylearn'],
    version='0.0.1',
    description='Tool for running remote machine learning jobs remotely.',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/alekseylearn',
    download_url='https://github.com/ResidentMario/alekseylearn/tarball/0.0.1',
    keywords=['data', 'machine learning', 'data engineering'],
    classifiers=[],
    include_package_data=True,
    entry_points='''
        [console_scripts]
        alekseylearn=alekseylearn.cli:cli
    ''',
)