import os
import numpy as np
import matplotlib.pyplot as plt
import pycbc
from pycbc import frame, psd

# First check if the file path is OK
frame_file = '/data/gravwav/twouters/datasets/G1124251/frames/L-L1_GWOSC_O4a_16KHZ_R1-1372278784-4096.gwf'
if not os.path.exists(frame_file):
    raise FileNotFoundError(f"Frame file {frame_file} does not exist.")

# data_l1_ch = frame.read_frame(frame_file, 'L1:GDS-CALIB_STRAIN_CLEAN_AR')
print(f"Reading frame file: {frame_file}")
data_l1_ch = frame.read_frame(frame_file, 'L1:GWOSC-16KHZ_R1_STRAIN')
print(f"Reading frame file: {frame_file} DONE")

print(f"Making PSD")
psd_l1_ch=data_l1_ch.psd(8)
print(f"Making PSD DONE")

print(f"Plotting PSD")
plt.loglog(psd_l1_ch.sample_frequencies, psd_l1_ch**0.5, label='L1-PSD')
plt.xlim(10, 4000)
plt.ylim(1e-24,1e-21)
plt.legend()
plt.savefig('psd_gwosc_type.png')
plt.close()
print(f"Plotting PSD DONE")

t_samples = psd_l1_ch.sample_frequencies.data
data_psd_l1 = np.array([psd_l1_ch.sample_frequencies.data[(t_samples<2048) & (t_samples>10)], psd_l1_ch.data[(t_samples<2048) & (t_samples>10)]]).T

# Save the PSD to the text file:
np.savetxt('psd_L1_G1124251_gwosc.txt', data_psd_l1)