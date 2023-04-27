# Homework 1

## Part 3

### i) Flat-field correction

I'll start by implementing this function as a simple function inside a file, maybe in the future it may be sensible to move it inside some sort of package?

To test the functionality, I used some data from [this](https://zenodo.org/record/2686726) dataset. I selected a few frames just to give a brief demonstration. If I understand correctly the files `di000000.tif` and `io000000.tif` are the dark frame and flat field respectively. The results seems satisfying.

![Flat field correction test](flatFieldCorrection.png "Flat field correction test")

