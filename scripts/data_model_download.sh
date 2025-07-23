#!/bin/bash

# Make directory if it doesn't exist
mkdir -p ../assets

# Download the OPV2V test split
echo "Downloading OPV2V test split..."
curl -L -o ../assets/opv2v_test_split.zip "https://arizona.box.com/shared/static/kd9gisu2xaiw7s4fpxzqpnag4md18zhd.zip" 

# Download the pre-trained model weights
echo "Downloading pre-trained model weights..."
curl -L -o ../assets/model_weights.zip "https://arizona.box.com/shared/static/kdcvfnqapcln931y7jjetw589sf81rtj.zip"

# Download the traffic jam dataset
echo "Downloading traffic jam dataset..."
curl -L -o ../assets/traffic_jam_dataset.zip "https://arizona.box.com/shared/static/wzgwvero68pp1t7ulx9ypt9xyj0r6r7l.zip"

# Unzip the files
echo "Unzipping files..."
unzip ../assets/opv2v_test_split.zip -d ../assets/
unzip ../assets/model_weights.zip -d ../assets/
unzip ../assets/traffic_jam_dataset.zip -d ../assets/

# Remove the zip files
echo "Removing zip files..."
rm ../assets/opv2v_test_split.zip
rm ../assets/model_weights.zip
rm ../assets/traffic_jam_dataset.zip

echo "Done!"
