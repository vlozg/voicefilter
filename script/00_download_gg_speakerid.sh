cd ../datasets/
mkdir ggspeaker_id
cd ggspeaker_id/

wget https://github.com/google/speaker-id/archive/refs/heads/master.zip \
    && (unzip -q master.zip \
        && mv speaker-id-master/publications/VoiceFilter/dataset/* ./ \
        && rm -rf speaker-id-master/ \
        && rm master.zip;) \
    || echo "Download google speaker ID dataset failed"

cd ../

# Get full path for the audios in tupples
python ./preprocess_ggspeakerid.py -b ./ggspeaker_id/