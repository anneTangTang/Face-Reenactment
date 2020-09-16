#!/bin/bash
# author: TangAnni

# Install requirements
pip install -r requirements.txt;

# Preprocess
# Convert video to frames
echo "Preprocessing...";
rm -rf target; mkdir target; mkdir target/{origin,recons,eye};
rm -rf source; mkdir source; mkdir source/origin;
ffmpeg -i "$1" -f image2 -qscale:v 2 source/origin/%05d.jpeg;
ffmpeg -i "$2" -f image2 -qscale:v 2 target/origin/%05d.jpeg;
# Detect face in the first frame and write the (x, y, w, h) position into the specified file.
cd preprocess;
python face_detection.py -i ../source/origin/00001.jpeg -o ../.cropsource;
python face_detection.py -i ../target/origin/00001.jpeg -o ../.croptarget;
# Crop frames according to params in the file and resize to 256 x 256
python crop_and_resize.py -i ../source/origin -f ../.cropsource;
python crop_and_resize.py -i ../target/origin -f ../.croptarget;

# Predict landmarks
echo "Predicting landmarks...";
cd ../face_landmarks;
python predictor.py -i ../source/origin -s ../source/landmark.json;
python predictor.py -i ../target/origin -s ../target/landmark.json;

# Estimate all parameters for target
echo "Estimating parameters of target...";
cd ../face3d/utils;
python estimate.py -i ../../target/origin -l ../../target/landmark.json -s ../../target/target.json;

# Render reconstructed images & eye images & mask for target
cd face3d/utils;
echo "Rendering reconstructed images...";
python render_target.py -i ../../target/target.json -s ../../target/recons;
echo "Rendering eye images...";
python render_eye.py -i ../../target/origin -l ../../target/landmark.json -s ../../target/eye;
echo "Rendering mask...";
python render_mask.py -i ../../target/target.json -s ../../target/mask.json;

# Training for target
echo "Training...";
cd ../../rendering2video;
python train.py --root ../target/;

# Estimate expression & pose parameters for source
echo "Estimating parameters of source...";
cd ../face3d/utils;
python estimate.py -i ../../source/origin -l ../../source/landmark.json -s ../../source/target.json --texture False;

# Drive target with source
echo "Driving...";
# Change parameters
rm -rf ../../drive; mkdir ../../drive; mkdir ../../drive/{recons,eye,generated};
python change_params.py -t ../../target/target.json -s ../../source/target.json --save ../../drive/drive.json;
# Render reconstructed images & eye images for drived target
python render_target.py -i ../../drive/drive.json -s ../../drive/recons;
cd ../../face_landmarks;
python predictor_drive.py -i ../drive/drive.json -s ../drive/landmark.json;
cd ../face3d/utils;
python render_eye_drive.py --source ../../source/origin --landmark_s ../../source/landmark.json \
--landmark_t ../../drive/landmark.json --save ../../drive/eye;
# Load pre-trained model and test
cd ../../rendering2video;
python test.py --root ../drive --target ../../target;
# Convert output frames to video
cd ../drive/generated;
ffmpeg -start_number 1 -i %05d.jpg -vf fps=25 ../drive.mp4;

echo "Success!";
