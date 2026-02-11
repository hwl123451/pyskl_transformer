#!/bin/bash

# UR Fall Detection Dataset Downloader
# Downloads RGB data files from the UR Fall Detection Dataset

# Base URL - you'll need to update this with the actual website URL
BASE_URL="https://fenix.ur.edu.pl/~mkepski/ds/"  # Replace with actual base URL

# Create directories for downloads
mkdir -p downloads/fall_rgb
mkdir -p downloads/adl_rgb

echo "Starting download of UR Fall Detection Dataset RGB files..."

# Download Fall RGB sequences (01-30)
echo "Downloading Fall RGB sequences..."
for i in $(seq -w 1 30); do
    for cam in 0 1; do
        filename="fall-${i}-cam${cam}-rgb.zip"
        url="${BASE_URL}/data/${filename}"

        echo "Downloading ${filename}..."
        wget -c -P downloads/fall_rgb/ "${url}" || {
            echo "Failed to download ${filename}"
            continue
        }
    done
done

# Download ADL RGB sequences (01-40, only cam0 available)
echo "Downloading ADL RGB sequences..."
for i in $(seq -w 1 40); do
    filename="adl-${i}-cam0-rgb.zip"
    url="${BASE_URL}/data/${filename}"

    echo "Downloading ${filename}..."
    wget -c -P downloads/adl_rgb/ "${url}" || {
        echo "Failed to download ${filename}"
        continue
    }
done

echo "Download complete!"
echo "Fall RGB files saved to: downloads/fall_rgb/"
echo "ADL RGB files saved to: downloads/adl_rgb/"

# Unzip all downloaded files
echo ""
echo "Unzipping all downloaded files in downloads/fall_rgb/..."
for zipfile in downloads/fall_rgb/*.zip; do
    echo "Unzipping $zipfile..."
    unzip -o "$zipfile" -d downloads/fall_rgb/
done

echo "Unzipping all downloaded files in downloads/adl_rgb/..."
for zipfile in downloads/adl_rgb/*.zip; do
    echo "Unzipping $zipfile..."
    unzip -o "$zipfile" -d downloads/adl_rgb/
done

# Optional: Display download summary
echo ""
echo "Download Summary:"
echo "Fall RGB files: $(ls downloads/fall_rgb/ | wc -l) files"
echo "ADL RGB files: $(ls downloads/adl_rgb/ | wc -l) files"
