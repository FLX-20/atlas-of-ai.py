import os
import re
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class PodcastScriptProcessor:

    MODEL = "playai-tts"

    def __init__(
        self,
        script_path: str,
        output_dir: str = "audio_output",
        voice_map: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
    ):
        self.script_path = Path(script_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.voice_map = voice_map or {}

        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No API key provided for Groq.")

        self.client = Groq(api_key=api_key)

        self.lines = self._read_script()
        self.speaker_names = self._extract_speaker_names()
        self.speaker_orders = self._extract_speaker_order()
        self.lines_per_speaker = self._extract_lines_per_speaker()

    def _read_script(self) -> List[str]:
        try:
            with self.script_path.open("r", encoding="utf-8") as file:
                return file.readlines()
        except FileNotFoundError:
            logging.error(f"Script file not found: {self.script_path}")
            raise

    def _extract_speaker_names(self) -> List[str]:
        name_pattern = re.compile(r'^\[([^\]]+)\]:')
        names = {match.group(1) for line in self.lines if (
            match := name_pattern.match(line.strip()))}
        return sorted(names)

    def _extract_speaker_order(self) -> List[str]:
        pattern = re.compile(
            r'^\[(' + '|'.join(re.escape(name) for name in self.speaker_names) + r')\]:')
        return [match.group(1) for line in self.lines if (match := pattern.match(line.strip()))]

    def _extract_lines_per_speaker(self) -> Dict[str, List[str]]:
        pattern = re.compile(r'^\[([^\]]+)\]:(.*)')
        speaker_lines = {name: [] for name in self.speaker_names}
        for line in self.lines:
            match = pattern.match(line.strip())
            if match:
                name, spoken_line = match.groups()
                speaker_lines[name].append(spoken_line.strip())
        return speaker_lines

    def generate_audio(self):
        """Generate audio files for each speaker line in script order, saving as numbered WAV files."""
        logging.info("Starting audio generation...")

        index_tracker = {speaker: 0 for speaker in self.speaker_names}

        for line_number, speaker in enumerate(self.speaker_orders, start=1):

            voice = self.voice_map.get(speaker)
            if not voice:
                logging.warning(
                    f"Skipping line {line_number}: No voice assigned for speaker '{speaker}'")
                continue

            speaker_lines = self.lines_per_speaker.get(speaker, [])
            speaker_index = index_tracker.get(speaker, 0)

            if speaker_index >= len(speaker_lines):
                logging.warning(
                    f"Line index out of bounds for speaker '{speaker}' at line {line_number}")
                continue

            text = speaker_lines[speaker_index]
            index_tracker[speaker] += 1

            filename = self._sanitize_filename(
                f"{line_number}_{speaker.lower()}.wav")
            try:
                self._generate_single_audio(text, filename, voice)
            except Exception as e:
                logging.exception(
                    f"Error generating audio for line {line_number} ({speaker}): {e}")

    logging.info("Audio generation completed.")

    def _generate_single_audio(self, text: str, filename: str, voice: str):
        logging.info(f"Generating audio: {filename}")
        speech_file_path = self.output_dir / filename
        response = self.client.audio.speech.create(
            model=self.MODEL,
            voice=voice,
            response_format="wav",
            input=text,
        )
        response.write_to_file(speech_file_path)

    def summary(self):
        logging.info(f"Speakers detected: {', '.join(self.speaker_names)}")
        for speaker in self.speaker_names:
            num_lines = len(self.lines_per_speaker.get(speaker, []))
            logging.info(f"  - {speaker}: {num_lines} line(s)")

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return re.sub(r'[^\w\-_.]', '_', name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate podcast audio from a script.")
    parser.add_argument("script_path", type=str, default="podcast_script.txt",
                        help="Path to the podcast script.")
    parser.add_argument("--output_dir", type=str, default="audio_output",
                        help="Output directory for audio files.")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Groq API key (or set GROQ_API_KEY env variable).")

    args = parser.parse_args()

    voice_mapping = {
        "Alan": "Atlas-PlayAI",
        "Arabella": "Deedee-PlayAI",
    }

    processor = PodcastScriptProcessor(
        script_path=args.script_path,
        output_dir=args.output_dir,
        voice_map=voice_mapping,
        api_key=args.api_key
    )

    processor.summary()
    processor.generate_audio()


if __name__ == "__main__":
    main()
