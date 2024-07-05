# Osmose-Pole-Diameter
Utility pole diameter estimation using images and AI for Osmose


## Usage:
%python src/main.py -i \<path-to-input-video\> -o \<path-to-output-video\> -d \<manually measured diameter in INCHES\>

%python src/main.py


## Output:
A sample of the result of AI estimation of diameter of the utility pole:
![frame0001](https://github.com/BrightDotAi/Osmose-Pole-Diameter/assets/29873946/c86556de-8d30-46ea-a088-604ad85d87ab)
The pole is annotated by a bounding-box and segmented region in blue. The badge is annotated by a bounding-box and segmented region in red.

The A.I. estimated diameter is shown in the lower-left portion of the result image. 
If a manually measured diameter is provided, then this manually measured diameter and estimated error is also shown in the resulting image.
