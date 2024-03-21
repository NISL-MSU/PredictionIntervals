import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PredictionIntervals',
    version='0.0.1',
    author='Giorgio Morales - Montana State University',
    author_email='giorgiomoralesluna@gmail.com',
    description='DualAQD: Dual Accuracy-quality-driven Prediction Intervals',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NISL-MSU/PredictionIntervals',
    project_urls={"Bug Tracker": "https://github.com/NISL-MSU/PredictionIntervals/issues"},
    license='MIT',
    package_dir={"": "src"},
    packages=setuptools.find_packages('src', exclude=['test']),
    install_requires=['matplotlib', 'numpy', 'opencv-python', 'sklearn', 'scipy', 'statsmodels', 'tqdm', 'timeout_decorator',
                      'h5py', 'pyodbc', 'regex', 'tensorboard', 'python-dotenv', 'omegaconf', 'pandas'],
)
