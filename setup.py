from setuptools import setup, find_packages


setup(name='Density Clustering',
      version='1.0.0',
      description='Bridging the Gap Between Supervised and Unsupervised Learning for Fine-grained Image Classification',
      author='Jiabao Wang',
      author_email='jiabao_1108@163.com',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu', 'hdbscan'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Fine-grained Classification'
      ])
