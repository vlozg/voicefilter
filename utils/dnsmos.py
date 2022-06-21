import os
import numpy as np
import onnxruntime as ort

from .DNSMOS.dnsmos_local import ComputeScore, INPUT_LENGTH, SAMPLING_RATE

'''
Wrapper for DNSMOS computation on the fly
'''
class DNSMOS(ComputeScore):
    def __init__(self, base_dir, is_personalized_MOS, use_cuda=True, sampling_rate=SAMPLING_RATE) -> None:
        if is_personalized_MOS:
            primary_model_path = os.path.join(base_dir, 'DNSMOS/pDNSMOS', 'sig_bak_ovr.onnx')
        else:
            primary_model_path = os.path.join(base_dir, 'DNSMOS/DNSMOS', 'sig_bak_ovr.onnx')
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        self.onnx_sess = ort.InferenceSession(primary_model_path, providers=providers)
        
        self.sr = sampling_rate
        self.len_samples = int(INPUT_LENGTH*self.sr)
        self.p_mos = is_personalized_MOS
        self.use_cuda = use_cuda

    def __call__(self, audio):
        actual_audio_len = len(audio)
        while len(audio) < self.len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/self.sr) - INPUT_LENGTH)+1
        hop_len_samples = self.sr
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < self.len_samples:
                continue
            
            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            input_features = ort.OrtValue.ortvalue_from_numpy(input_features)
            oi = {'input_1': input_features}

            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run_with_ort_values(None, oi)[0].numpy()[0]
            
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,self.p_mos)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {'len_in_sec': actual_audio_len/self.sr}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        return clip_dict