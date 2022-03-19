apt install axel -y
cd ../datasets

# Download, then extract LibriSpeech datasets
(axel -n 10 -q -c "https://www.openslr.org/resources/12/train-clean-100.tar.gz"\
    && tar -xzf train-clean-100.tar.gz\
    && rm train-clean-100.tar.gz) &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/train-clean-360.tar.gz"\
    && tar -xzf train-clean-360.tar.gz\
    && rm train-clean-360.tar.gz) &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/dev-clean.tar.gz"\
    && tar -xzf dev-clean.tar.gz\
    && rm dev-clean.tar.gz) &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/test-clean.tar.gz"\
    && tar -xzf test-clean.tar.gz\
    && rm test-clean.tar.gz) &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/dev-other.tar.gz"\
    && tar -xzf dev-other.tar.gz\
    && rm dev-other.tar.gz) &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/test-other.tar.gz"\
    && tar -xzf test-other.tar.gz\
    && rm test-other.tar.gz) &

# Download VCTK dataset
mkdir VCTK
cd VCTK/
(axel -n 10 -q -c "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"\
    && unzip -q DS_10283_3443.zip\
    && rm DS_10283_3443.zip) &
cd ../

wait