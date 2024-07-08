# Osmose-Pole-Diameter
Utility pole diameter estimation using images and AI for Osmose

## Mac Users

### Install Homebrew
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

OR

install using homebrew package: https://github.com/Homebrew/brew/releases/download/4.3.9/Homebrew-4.3.9.pkg

$ wget https://github.com/Homebrew/brew/releases/download/4.3.9/Homebrew-4.3.9.pkg

$ sud installer -pkg Homebrew-4.3.9.pkg


### Install ffmpeg

$ brew install ffmpeg


### Install pipx

$ brew instal pipx


### Instll poetry
If you haven't already, install Poetry. You can install it via the following command:

$ pipx install poetry


### Clone the project from Github

$ git clone \<project-git-link\>

$ cd "Osmose-Pole-Diameter"

### Install Project dependencies using Poetry

Once Poetry is installed and you are in the cloned repository's directory, install the project dependencies:

$ poetry install


## Usage

### Using Poetry shell
Poetry manages a virtual environment for your project. You can activate it using:

$ poetry shell

(\<poetry-venv\>)$ python src/main.py -i \<path-to-input-video\> -o \<path-to-output-video\> -d \<manually measured diameter in INCHES\>

(\<poetry-venv\>)$ python src/main.py

(\<poetry-venv\>)$ python src/main.py -d 9.5

(\<poetry-venv\>)$ python src/main.py -i \<path-to-input-video\> -d \<manually measured diameter in INCHES\>

### Running file directly

$ poetry run python src/main.py -i \<path-to-input-video\> -o \<path-to-output-video\> -d \<manually measured diameter in INCHES\>

$ poetry run python src/main.py

$ poetry run python src/main.py -d 9.5

$ poetry run python src/main.py -i \<path-to-input-video\> -d \<manually measured diameter in INCHES\>


## Result

A sample of the result of AI estimation of diameter of the utility pole:

![frame0001](https://github.com/BrightDotAi/Osmose-Pole-Diameter/assets/29873946/c86556de-8d30-46ea-a088-604ad85d87ab)

The pole is annotated by a bounding-box and segmented region in blue. The badge is annotated by a bounding-box and segmented region in red.

The A.I. estimated diameter is shown in the lower-left portion of the result image. 

If a manually measured diameter is provided, then this manually measured diameter and estimated error is also shown in the resulting image.
