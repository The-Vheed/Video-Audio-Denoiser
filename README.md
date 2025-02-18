# Video Audio Denoiser

A Python-based tool that extracts audio from video files, performs noise reduction using DeepFilterNet, and then reattaches the enhanced audio back to the video. This tool leverages state-of-the-art deep learning models (via PyTorch and DeepFilterNet) and requires audio files to be at 48 kHz for optimal performance.

## Features

- **Audio Extraction:** Extracts audio from MP4 video files.
- **Resampling:** Uses [librosa](https://librosa.org/) and [soundfile](https://pysoundfile.readthedocs.io/) to ensure audio is resampled to 48 kHz.
- **Deep Learning Denoising:** Enhances noisy audio using DeepFilterNet, a deep learning–based noise removal model.
- **Video Reassembly:** Reattaches the enhanced audio back to the original video file.

## Prerequisites

Before installing the Python dependencies, make sure you have the following installed:

### 1. PyTorch

Install PyTorch (with CPU support or CUDA if your system supports it). For CPU-only support, run:

```bash
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For CUDA support, please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the appropriate installation command.

### 2. Rust Toolchain

DeepFilterNet depends on Rust for building native extensions. Install Rust via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installation, ensure that `cargo` is in your system PATH.

### 3. Python Dependencies

Install the remaining Python packages by running:

```bash
pip install -r requirements.txt
```

Where `requirements.txt` should include (but is not limited to):

```
moviepy
librosa
soundfile
deepfilternet
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vheed/video-audio-denoiser.git
   cd video-audio-denoiser
   ```

2. **Install PyTorch:**

   ```bash
   pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install Rust:**

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

4. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main script extracts the audio from a given video, resamples it if necessary, enhances the audio using DeepFilterNet, and then reassembles the video with the improved audio.

Edit the input and output file paths as needed in the script, then run:

```bash
python3 process_video.py input.mp4 output.mp4
```

*Example:*  
If your input video is named `input.mp4` and you want the output to be `output.mp4`, ensure the script reflects these names.

## Troubleshooting

- **Rust/Maturin Issues:**  
  If you encounter errors related to Rust or maturin (e.g., updating the crates.io index or SSL errors), ensure:
  - Rust is correctly installed and updated.
  - Your network or proxy settings allow cargo to access `https://github.com/rust-lang/crates.io-index`.
  - You can try setting the environment variable to use the Git CLI for fetching dependencies:
    ```bash
    export CARGO_NET_GIT_FETCH_WITH_CLI=true
    ```

- **Dependency Installation:**  
  Ensure that pip, setuptools, and wheel are up to date:
  ```bash
  pip install --upgrade pip setuptools wheel
  ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes. Also, a star and a follow would be appreciated!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
