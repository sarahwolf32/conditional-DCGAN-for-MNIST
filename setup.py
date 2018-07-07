from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=1.7']

# To train on GCP:
# gcloud ml-engine jobs submit training job_47 --package-path trainer --module-name trainer.task --scale-tier basic-gpu --region us-central1 --staging-bucket gs://gan-training-207705_bucket2 

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    author='Sarah Wolf',
    description='Generating MNIST characters with a cDCGAN'
)