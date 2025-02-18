import os
import argparse
import tempfile
import librosa
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip
from df.enhance import enhance, init_df, load_audio, save_audio


def resample_to_48k(audio_path):
    """
    Ensures the audio file is sampled at 48 kHz.
    If not, it loads the audio and resamples it to 48kHz.
    """
    print("Resampling audio to 48kHz if needed...")
    audio, orig_sr = librosa.load(audio_path, sr=None)
    if orig_sr != 48000:
        audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=48000)
        sf.write(audio_path, audio_resampled, 48000)
        print(f"Audio resampled from {orig_sr} Hz to 48000 Hz.")
    else:
        print("Audio is already at 48kHz.")


def enhance_audio_deepfilternet(input_audio_path, output_audio_path):
    """
    Uses DeepFilterNet to enhance (denoise) the audio.
    DeepFilterNet requires 48kHz audio.
    """
    print("Initializing DeepFilterNet model...")
    model, df_state, _ = init_df()

    print("Loading audio for enhancement...")
    # load_audio will load the audio at the sampling rate expected by the model (48kHz)
    audio, _ = load_audio(input_audio_path, sr=df_state.sr())

    print("Enhancing audio with DeepFilterNet...")
    enhanced = enhance(model, df_state, audio)

    print("Saving enhanced audio...")
    save_audio(output_audio_path, enhanced, df_state.sr())
    print("Audio enhancement complete.")


def process_video(input_video_path, output_video_path):
    # Load the input video
    print("Loading video...")
    clip = VideoFileClip(input_video_path)

    # Extract audio to a temporary WAV file.
    # We request a sample rate of 48000 so that the file is (or should be) in the correct format.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    print("Extracting audio from video...")
    clip.audio.write_audiofile(temp_audio_path, fps=48000, nbytes=2, codec="pcm_s16le")

    # Ensure the audio is 48kHz (resample if necessary)
    resample_to_48k(temp_audio_path)

    # Enhance the audio using DeepFilterNet
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_enhanced_file:
        temp_enhanced_audio_path = temp_enhanced_file.name
    enhance_audio_deepfilternet(temp_audio_path, temp_enhanced_audio_path)

    # Attach the enhanced audio to the original video
    print("Attaching enhanced audio to video...")
    new_audio_clip = AudioFileClip(temp_enhanced_audio_path)
    new_clip = clip.with_audio(new_audio_clip)

    # Write the final video file with the new audio
    print("Writing final video file...")
    new_clip.write_videofile(output_video_path, audio_codec="aac")

    # Clean up temporary files
    os.remove(temp_audio_path)
    os.remove(temp_enhanced_audio_path)
    print("Temporary files removed. Process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Audio Denoiser")
    parser.add_argument(
        "input_video", help="Path to the input video file", default="input_video.mp4"
    )
    parser.add_argument(
        "output_video",
        help="Path for the output video file",
        default="output_video.mp4",
    )
    args = parser.parse_args()

    process_video(args.input_video, args.output_video)
