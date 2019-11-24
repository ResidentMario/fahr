import fahr
import os
import shutil

# export FAHR_KAGGLE_USERNAME=ResidentMario
shutil.copy('../fixtures/train_kaggle/train.py', 'train.py')
j = fahr.TrainJob(
    'train.py', train_driver='kaggle',
    config={'username': os.environ.get('FAHR_KAGGLE_USERNAME')},
    overwrite=True
)
j.train()
