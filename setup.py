from distutils.core import setup

setup(
    name='ml_models',
    version='0.1',
    packages=['cnn_models', 'cnn_models.models', 'cnn_models.iterators', 'cnn_models.cifar_models',
              'cnn_models.nr_iqa_models'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='filip141',
    author_email='201134@student.pwr.wroc.pl',
    description='', requires=['numpy', 'cv2', 'tensorflow_gpu']
)
