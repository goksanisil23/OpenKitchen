Runs through all the race tracks available under the given folder and generates: 

Measurement mode is selected via kMeasurementMode:
A) 2d pointcloud measurement from robot at that step + (velocity + delta_steering) action
B) Birdseye image view from robot at that step + (velocity + delta_steering) action

Usage:
./collect_data_racetracks path_to_racetracks/racetrack-database/tracks/

A specific track can be selected by specifying kTrackName, if empty all tracks will be used

Control inputs are bounded to (-+kSteeringAngleClampDeg) degrees for steering, and [0, 100] for throttle

Pointclouds or images will be saved under measurements_and_actions folder

Format
A) Pointcloud
- First N rows are 2d laser measurements
- Last row is the action:  Robot speed, steering delta
B) Birdseye image
- birdseye_TRACK_N.png (birdseye view RGB image)
- birdseye_TRACK_N.txt (action in 1 line)
