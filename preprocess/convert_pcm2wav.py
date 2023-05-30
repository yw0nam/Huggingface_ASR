"""
This script converts pcm audio files to wav audio files.

Usage:
python convert_pcm2wav.py --root_dir=<path/to/root/dir> --out_dir=<path/to/out/dir>

Arguments:
--root_dir: str, required
Root directory containing pcm audio files to convert to wav.

Returns:
None
"""
from tqdm import tqdm
import argparse, os
from glob import glob
import numpy as np
import soundfile

def define_argparser():
    """
    Define command line argument parser.
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()

    p.add_argument("--root_dir", type=str, required=True)
    config = p.parse_args()

    return config

def main(config):
    """
    Convert pcm audio files to wav audio files.

    Args:
        config (argparse.Namespace): Command line arguments

    Returns:
        None
    """
    data_pathes = sorted(glob(os.path.join(config.root_dir, '*/*/*.pcm')))
    # data_pathes = sorted(glob(os.path.join(config.root_dir, '*.pcm')))
    print("\nConverting Start")
    
    for path in tqdm(data_pathes):
        out_path = os.path.join(os.path.dirname(path),os.path.basename(path)[:-4] + '.wav')
        data_type = np.dtype('i2')  # 16-bit signed integer
        data_size = os.path.getsize(path)  # get the size of the file in bytes
        data_len = data_size // data_type.itemsize
        # pad the file with zeros if necessary
        if data_size % data_type.itemsize != 0:
            pad_size = data_type.itemsize - data_size % data_type.itemsize
            with open(path, 'ab') as f:
                f.write(b'\x00' * pad_size)
        data = np.memmap(path, dtype=data_type, mode='r', shape=(data_len,))
        soundfile.write(out_path, data, 16000)
        
if __name__ == "__main__":
    config = define_argparser()
    main(config)
    