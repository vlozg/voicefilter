cd ../utils/

(wget -c https://github.com/microsoft/DNS-Challenge/archive/refs/heads/master.zip \
    && { unzip -q master.zip && rm master.zip \
        && mv DNS-Challenge-master/DNSMOS . && rm -rf DNS-Challenge-master; } \
    || echo "Fail to download DNS-Challange repo for DNSMOS")