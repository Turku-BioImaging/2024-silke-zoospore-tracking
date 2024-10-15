# Zoospore Tracking


## Experiment Protocol
Light intensity levels are changed many times during a single experimental run, with phases (step up, step down, and adaptation). Numbers in brackets are codes for the light level.
![image](https://github.com/Turku-BioImaging/project-silke-zoospore-tracking/assets/11444749/e32b3513-d502-4ef9-b054-af4afdb87987)

## Possible metrics
![image](https://github.com/Turku-BioImaging/project-silke-zoospore-tracking/assets/11444749/f4a37da3-6dd1-4d3c-b69f-8b2b37505d2c)

## Implemented metrics
- __Frequency of direction change__. Angular velocity vector, with threshold for turn angle directional change.
- __Mean Squared Displacement (MSD)__.
- __Straight-line velocity (VSL)__.  The time-average velocity of the zoospore object along a straight line between its first detected position and its last position.
- __Curvilinear velocity (VCL)__. The time-average velocity of the object along its actual trajectory.

## Execute image processing

This runs the entire image processing pipeline.
```
cd ./src
python main.py --object-detection \
    --linking \
    --metrics
```
