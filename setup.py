import setuptools
import malaya_boilerplate

__packagename__ = 'malaya-boilerplate'


with open('requirements.txt') as fopen:
    req = list(filter(None, fopen.read().split('\n')))

setuptools.setup(
    name = __packagename__,
    packages = setuptools.find_packages(),
    version = malaya_boilerplate.__version__,
    python_requires = '>=3.6.*',
    description = 'Tensorflow freeze graph optimization and boilerplates to share among Malaya projects.',
    author = 'huseinzol05',
    author_email = 'husein.zol05@gmail.com',
    url = 'https://github.com/huseinzol05/malaya-boilerplate',
    download_url = 'https://github.com/huseinzol05/malaya-boilerplate/archive/master.zip',
    keywords = ['nlp', 'bm'],
    install_requires = req,
    license = 'MIT',
    classifiers = [
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
