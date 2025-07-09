#!/bin/bash

# Make directory if it doesn't exist
mkdir -p ../assets

# Download the OPV2V test split
echo "Downloading OPV2V test split..."
curl -L -o ../assets/opv2v_test_split.zip "https://arizona.box.com/shared/static/kd9gisu2xaiw7s4fpxzqpnag4md18zhd.zip" 

# Download the pre-trained model weights
echo "Downloading pre-trained model weights..."
curl -L -o ../assets/model_weights.zip "https://arizona.box.com/shared/static/kdcvfnqapcln931y7jjetw589sf81rtj.zip"

# Unzip the files
echo "Unzipping files..."
unzip ../assets/opv2v_test_split.zip -d ../assets/
unzip ../assets/model_weights.zip -d ../assets/

# Remove the zip files
echo "Removing zip files..."
rm ../assets/opv2v_test_split.zip
rm ../assets/model_weights.zip

echo "Done!"
