cd ../datasets/
mkdir VCTK
cd VCTK/

# Download VCTK dataset

# Does not support fast download
# Also some audio files in Google Speaker ID missed in this dataset so ...
# (axel -a -n 10 -c "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y" \
#     && { unzip -q -n VCTK-Corpus-0.92 \
#     && mv wav48_silence_trimmed wav48 \
#     && rm VCTK-Corpus-0.92.zip; } \

(axel -a -n 10 -c "http://www.udialogue.org/download/VCTK-Corpus.tar.gz" \
    && { tar -xzf VCTK-Corpus.tar.gz \
    && rm VCTK-Corpus.tar.gz \
    && mv VCTK-Corpus/* ./
    && rm -rf VCTK-Corpus; } \
    || echo "Download VCTK dataset failed") 
cd ../

# Preprocess, convert 48k to 16k
python convert_job.py -j to16khz -d VCTK/ \
    && python convert_job.py -j remove16kprefix -d VCTK