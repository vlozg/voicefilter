cd ../datasets

# Download, then extract LibriSpeech datasets
(axel -n 10 -q -c "https://www.openslr.org/resources/12/train-clean-100.tar.gz" \
    && { tar -xzf train-clean-100.tar.gz && rm train-clean-100.tar.gz; } \
    || echo "Download train-clean-100 failed") &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/train-clean-360.tar.gz" \
    && { tar -xzf train-clean-360.tar.gz && rm train-clean-360.tar.gz; } \
    || echo "Download train-clean-360 failed") &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/dev-clean.tar.gz" \
    && { tar -xzf dev-clean.tar.gz && rm dev-clean.tar.gz; } \
    || echo "Download dev-clean failed") &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/test-clean.tar.gz" \
    && { tar -xzf test-clean.tar.gz && rm test-clean.tar.gz; } \
    || echo "Download test-clean failed") &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/dev-other.tar.gz" \
    && { tar -xzf dev-other.tar.gz && rm dev-other.tar.gz; } \
    || echo "Download dev-other failed") &
(axel -n 10 -q -c "https://www.openslr.org/resources/12/test-other.tar.gz" \
    && { tar -xzf test-other.tar.gz && rm test-other.tar.gz; } \
    || echo "Download test-other failed") &


wait
