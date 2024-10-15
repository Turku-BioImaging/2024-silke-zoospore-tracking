# Zoospore Tracking

## Experiment Protocol

Light intensity levels are changed many times during a single experimental run, with phases (step up, step down, and adaptation). Numbers in brackets are codes for the light level.
![image](https://github.com/Turku-BioImaging/project-silke-zoospore-tracking/assets/11444749/e32b3513-d502-4ef9-b054-af4afdb87987)

## Possible metrics

![image](https://github.com/Turku-BioImaging/project-silke-zoospore-tracking/assets/11444749/f4a37da3-6dd1-4d3c-b69f-8b2b37505d2c)

## Implemented metrics

- **Frequency of direction change**. Angular velocity vector, with threshold for turn angle directional change.
- **Mean Squared Displacement (MSD)**.
- **Straight-line velocity (VSL)**. The time-average velocity of the zoospore object along a straight line between its first detected position and its last position.
- **Curvilinear velocity (VCL)**. The time-average velocity of the object along its actual trajectory.
- **Directionality ratio**. A measure of how straight the overall path of movement is, defined as the ratio of the net displacement to the total path length.
- **Area covered**. The spatial extent of the area explored by the organism.
- **Average speed**. The mean speed of a particle of the entire trajectory.

## Execute image processing

This runs the entire image processing pipeline.

```
cd ./src
python main.py --object-detection \
    --linking \
    --metrics
```
