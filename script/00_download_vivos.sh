cd ../datasets/
mkdir VIVOS
cd VIVOS/


(gdown 1TU1FkVWruc16uG80YUVFLaAeGTmwd86- \
    && { tar -xvf vivos.tar.gz \
    && rm vivos.tar.gz; } \
    || echo "Download VIVOS dataset failed")
cd ../

