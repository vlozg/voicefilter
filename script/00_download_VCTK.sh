cd ../datasets/
mkdir VCTK
cd VCTK/

# Download VCTK dataset
(axel -a -c "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y" \
    && { unzip -q VCTK-Corpus-0.92 \
    && mv wav48_silence_trimmed wav48 \
    && rm VCTK-Corpus-0.92.zip; } \
    || echo "Download VCTK dataset failed") 
cd ../
