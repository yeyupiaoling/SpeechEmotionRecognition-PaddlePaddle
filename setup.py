from setuptools import setup, find_packages

import ppser

VERSION = ppser.__version__


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_requirements():
    with open('./requirements.txt', encoding="utf-8") as f:
        requirements = f.readlines()
    return requirements


if __name__ == "__main__":
    setup(
        name='ppser',
        packages=find_packages(),
        author='yeyupiaoling',
        version=VERSION,
        install_requires=parse_requirements(),
        description='Speech Emotion Recognition toolkit on PaddlePaddle',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='https://github.com/yeyupiaoling/SpeechEmotionRecognition-PaddlePaddle',
        download_url='https://github.com/yeyupiaoling/SpeechEmotionRecognition-PaddlePaddle.git',
        keywords=['audio', 'paddlepaddle', 'emotion'],
        classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Natural Language :: Chinese (Simplified)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9', 'Topic :: Utilities'
        ],
        license='Apache License 2.0',
        ext_modules=[])
