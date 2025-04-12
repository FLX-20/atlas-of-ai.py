import os
import re
from typing import List, Dict
from pydub import AudioSegment
from collections import defaultdict


class AudioConcatenator:
    def __init__(self, audio_dir: str, speaker_order: List[str] = None):
        self.audio_dir = audio_dir
        self.speaker_order = speaker_order
        self.pattern = re.compile(r"(\d+)_([a-zA-Z]+)\.wav")
        self.audio_groups: Dict[int,
                                Dict[str, AudioSegment]] = defaultdict(dict)

    def load_audio_files(self) -> None:
        print(f"[INFO] Loading audio files from: {self.audio_dir}")
        for filename in sorted(os.listdir(self.audio_dir)):
            match = self.pattern.match(filename)
            if match:
                index, speaker = int(match.group(1)), match.group(2).lower()
                file_path = os.path.join(self.audio_dir, filename)
                try:
                    audio = AudioSegment.from_wav(file_path)
                    self.audio_groups[index][speaker] = audio
                    print(f"[INFO] Loaded: {filename}")
                except Exception as e:
                    print(f"[WARNING] Could not load {filename}: {e}")

    def concatenate_audio(self) -> AudioSegment:
        print("[INFO] Concatenating audio segments...")
        combined = AudioSegment.empty()
        for index in sorted(self.audio_groups.keys()):
            for speaker, audio in self.audio_groups[index].items():
                combined += audio
                print(f"[DEBUG] Added index {index} - speaker {speaker}")
        return combined

    def export_final_audio(self, output_filename: str = "final_output.wav") -> None:
        self.load_audio_files()
        combined_audio = self.concatenate_audio()

        output_path = os.path.join(self.audio_dir, output_filename)
        combined_audio.export(output_path, format="wav")
        print(f"[SUCCESS] Final audio exported to: {output_path}")


if __name__ == "__main__":
    speakers = ["alan", "arabella"]

    audio_path = "audio_output"
    concatenator = AudioConcatenator(audio_path, speaker_order=speakers)
    concatenator.export_final_audio()
