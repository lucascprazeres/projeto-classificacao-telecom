import numpy as np
from commpy.channels import SISOFlatChannel
from ml4comm.qam_crazy import crazy_channel_propagate

def setup_channel(channel_type, symbs, code_rate, Es, SNR_dB):
    if channel_type == 'awgn':
        channel = SISOFlatChannel(None, (1 + 0j, 0j))
        channel.set_SNR_dB(SNR_dB, float(code_rate), Es)
        channel_output = channel.propagate(symbs)
    elif channel_type == 'crazy':
        channel_output = crazy_channel_propagate(symbs, SNR_dB) 
    else:
        raise ValueError(f'Channel type {channel_type} not found')
    return channel_output

def generate_dataset(channel_output, indexes):
  # Train
  train_size = int(len(indexes)-3000)
  y_train = indexes[:train_size]
  X_train = np.stack([np.real(channel_output[:train_size]),
                      np.imag(channel_output[:train_size])], axis=1)

  # Test
  y_test = indexes[train_size:]
  X_test = np.stack([np.real(channel_output[train_size:]),
                    np.imag(channel_output[train_size:])], axis=1)
  return [X_train, X_test, y_train, y_test]