screen -dmS VCTK_DOWNLOAD
screen -S VCTK_DOWNLOAD -X stuff "./00_download_VCTK.sh^M"


screen -dmS LibriSpeech_DOWNLOAD
screen -S LibriSpeech_DOWNLOAD -X stuff "./00_download_librispeech.sh^M"



screen -dmS VoxCeleb1_DOWNLOAD
screen -S VoxCeleb1_DOWNLOAD -X stuff "./00_download_voxceleb1.sh^M"

screen -dmS Zalo_DOWNLOAD
screen -S Zalo_DOWNLOAD -X stuff "./00_download_ZaloAI.sh^M"


screen -dmS Vin_DOWNLOAD
screen -S Vin_DOWNLOAD -X stuff "./00_download_Vin.sh^M"