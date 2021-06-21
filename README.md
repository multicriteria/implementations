# implementations
Implementations for the experimental evaluation of multi-criteria graph drawing

## Development

This application requires python with some packages which depends on UNIX. This files assumes that you run in an UNIX like environment. Currently it also seems to work as well in the Windows Subsystem for Linux.

### Init project workspace

First install python, pip, pipenv

```bash
sudo apt update
sudo apt install -y python3 python3-pip                 # installs the tools to run the application 
echo "export PATH=\"~/.local/bin:\$PATH\"" >> ~/.bashrc # add pip installed binaries to path
pip3 install pipenv                                     # installs pipenv
exec bash                                               # load the new PATH variable
python3 --version                                       # should show at least python 3.6 or higher
pipenv --version                                        # should show something like pipenv, version 2018.11.26
```
To run the implementation we have to install some required packages

```
pip3 install numpy
pip3 install networkx
pip3 install matplotlib
pip3 install math
pip3 install random
pip3 install copy
pip3 install time
pip3 install torch torchvision torchaudio
pip3 install rtree
```

#### Production

1. Get the sources to the target server by checking out this repository on the target server via `git clone https://github.com/multicriteria/implementations.git && cd implementations`
2. run main.py

You can change the output drawing by altering the pos_o in line 63 to any other pos that you want the drawing of. For example replace it with 'pos_dh' for the Davidson & Harel algorithm.
