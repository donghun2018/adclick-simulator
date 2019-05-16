# adclick-simulator

This is a fully agent-based sponsored search auction (SSA) simulator.
The simulator takes multiple bidding policies as agents participating in the auctions, and emulates 2nd price auction for sponsored search advertisement placing similar to Google's Ad-click system.

This simulator has been utilized in the following publications and educational projects:
- Optimal Learning (ORF 418) class offered in Princeton University. Spring 2018 semester. [Course Webpage](https://castlelab.princeton.edu/orf-418/)
- Meta-learning of Bidding Agent with Knowledge Gradientin a Fully Agent-based Sponsored Search Auction Simulator, AAMAS 2019 [PDF](http://www.ifaamas.org/Proceedings/aamas2019/pdfs/p2090.pdf)

## Get the Newest Code

Downloadable zip and tar.gz files are [here](https://github.com/donghun2018/adclick-simulator/releases). Also, you may clone this repository.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This software was tested with Python 3.6 running on Windows 10 and Ubuntu 16.04.

#### Python

Anaconda is an easy way to get a working python.
Get it [here](https://www.anaconda.com/download/).

The initial version of the simulator is tested on 64-bit python 3.6 as follows:

```
Python 3.6.1 | packaged by conda-forge | (default, May 23 2017, 14:21:39) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

#### Packages

The packages required by the simulator are:

- numpy: used for pseudorandom number generator and many useful functions
- openpyxl: used for Microsoft Excel xlsx input/output functionalities

You may get these using conda

```
$ conda install numpy openpyxl
```

or using pip

```
$ pip install numpy openpyxl
```

### Installing

Download the source codes by

```
$ git clone https://github.com/donghun2018/adclick-simulator.git
```

And try running

```
$ python simulator
```

This should generate several .xlsx files in the same directory.
Now you have a running simulator, and you may add your policies by adding a new policy file in Policies/ subdirectory.

### Other documents

- An introductory slideshow PDF file is available [here](https://github.com/donghun2018/adclick-simulator/blob/master//documentation/20180411_Ad-click_simulator_intro_r1.pdf)
  - This material is used as a guide to a Princeton undergraduate class offered in 2018.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/donghun2018/adclick-simulator/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning and Changelog

Please see [VERSION.md](https://github.com/donghun2018/adclick-simulator/blob/master/VERSION.md) for details.

## Authors

- Donghun Lee (d.lee at princeton dot edu): created *initial version*

See also the list of [contributors](https://github.com/donghun2018/adclick-simulator/contributors) who participated in this project.

## Acknowledgments

- Weidong Han for many useful feedback, suggestions, and contributions to improve the software

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/donghun2018/adclick-simulator/blob/master/LICENSE) file for details.
