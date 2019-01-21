from setuptools import setup
setup(
    name='alekseylearn',
    packages=['alekseylearn'], # this must be the same as the name above
    install_requires=['docker', 'jinja2'],
    py_modules=['alekseylearn'],
    version='0.0.1',
    description='Tools for running remote machine learning jobs.',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/alekseylearn',
    download_url='https://github.com/ResidentMario/alekseylearn/tarball/0.0.1',
    keywords=['data', 'machine learning', 'data engineering'],
    classifiers=[],
    include_package_data=True
)