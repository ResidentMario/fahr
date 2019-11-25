import fahr
import os
import shutil

# export FAHR_SAGEMAKER_ROLE_NAME=fahr-test-role
# export FAHR_SAGEMAKER_OUTPUT_PATH=s3://fahr-test/
shutil.copy('../fixtures/train_sagemaker/train.py', 'train.py')
shutil.copy('../fixtures/train_sagemaker/environment.yaml', 'environment.yaml')
j = fahr.TrainJob(
    'train.py', train_driver='sagemaker', train_image='default-cpu',
    config={
        'role_name': os.environ.get('FAHR_SAGEMAKER_ROLE_NAME'),
        'output_path': os.environ.get('FAHR_SAGEMAKER_OUTPUT_PATH')
    },
    envfile='environment.yaml',
    overwrite=True
)
j.build()
j.push()
j.train()
