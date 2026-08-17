import os
import threading
import wave
from datetime import datetime
from enum import Enum, auto

import numpy as np
import sounddevice as sd
from openwakeword import VAD
from openwakeword.model import Model
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Silero VAD inside openwakeword expects 480-sample frames (30 ms at 16 kHz)
_VAD_FRAME = 480


class _State(Enum):
    LISTENING = auto()
    RECORDING = auto()


class WakeWordNode(Node):
    def __init__(self):
        super().__init__('wake_word_node')

        self.declare_parameter('model_paths', [''])
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('chunk_size', 1280)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('cooldown_s', 1.0)
        self.declare_parameter('output_dir', os.path.expanduser('~/nestor_recordings'))
        self.declare_parameter('vad_speech_threshold', 0.5)
        self.declare_parameter('vad_silence_duration_s', 0.8)
        self.declare_parameter('max_utterance_duration_s', 10.0)

        model_paths = self.get_parameter('model_paths').get_parameter_value().string_array_value
        self._threshold = self.get_parameter('threshold').get_parameter_value().double_value
        self._chunk_size = self.get_parameter('chunk_size').get_parameter_value().integer_value
        self._sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        cooldown_s = self.get_parameter('cooldown_s').get_parameter_value().double_value
        self._output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self._vad_threshold = self.get_parameter('vad_speech_threshold').get_parameter_value().double_value
        vad_silence_s = self.get_parameter('vad_silence_duration_s').get_parameter_value().double_value
        max_utt_s = self.get_parameter('max_utterance_duration_s').get_parameter_value().double_value

        os.makedirs(self._output_dir, exist_ok=True)

        valid_paths = [p for p in model_paths if p]
        if valid_paths:
            self.get_logger().info(f'Loading custom models: {valid_paths}')
            self._oww = Model(wakeword_model_paths=valid_paths)
        else:
            self.get_logger().info('Loading default pre-trained models')
            self._oww = Model()

        self._vad = VAD()

        self._cooldown_ticks = int(cooldown_s * self._sample_rate / self._chunk_size)
        self._cooldown_remaining: dict[str, int] = {}

        # How many consecutive silent VAD frames end the utterance
        self._silence_frames_threshold = int(vad_silence_s * self._sample_rate / _VAD_FRAME)
        # Safety cap: max number of VAD frames in one utterance
        self._max_vad_frames = int(max_utt_s * self._sample_rate / _VAD_FRAME)

        self._state = _State.LISTENING
        self._rec_buf: list[np.ndarray] = []
        self._speech_started = False
        self._silence_count = 0
        self._total_vad_frames = 0
        # Carry-over samples that don't fill a full VAD frame yet
        self._vad_carry: np.ndarray = np.empty(0, dtype=np.int16)

        self._wake_pub = self.create_publisher(String, 'stt/wake_word_detected', 10)
        self._file_pub = self.create_publisher(String, 'stt/utterance_file', 10)

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self._chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self.get_logger().info('Wake word node ready, listening for wake words')

    # ------------------------------------------------------------------
    # Audio callback (runs in sounddevice thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            self.get_logger().warn(f'Audio stream status: {status}')

        audio = indata[:, 0].copy()

        if self._state is _State.LISTENING:
            self._handle_listening(audio)
        else:
            self._handle_recording(audio)

    def _handle_listening(self, audio: np.ndarray) -> None:
        predictions = self._oww.predict(audio)

        for word, score in predictions.items():
            remaining = self._cooldown_remaining.get(word, 0)
            if remaining > 0:
                self._cooldown_remaining[word] = remaining - 1
                continue

            if score >= self._threshold:
                self.get_logger().info(f'Wake word detected: "{word}" (score={score:.3f})')
                self._wake_pub.publish(String(data=word))
                self._cooldown_remaining[word] = self._cooldown_ticks
                self._start_recording()
                # Include this chunk so speech right after the wake word is captured
                self._handle_recording(audio)
                return  # only trigger once per callback

    def _start_recording(self) -> None:
        self._state = _State.RECORDING
        self._rec_buf = []
        self._speech_started = False
        self._silence_count = 0
        self._total_vad_frames = 0
        self._vad_carry = np.empty(0, dtype=np.int16)
        self._vad.reset_states()
        self.get_logger().info('Recording utterance...')

    def _handle_recording(self, audio: np.ndarray) -> None:
        self._rec_buf.append(audio)

        combined = np.concatenate([self._vad_carry, audio])
        n_frames = len(combined) // _VAD_FRAME
        self._vad_carry = combined[n_frames * _VAD_FRAME:]

        for i in range(n_frames):
            frame = combined[i * _VAD_FRAME:(i + 1) * _VAD_FRAME]
            speech_prob = self._vad.predict(frame)
            self._total_vad_frames += 1

            if speech_prob >= self._vad_threshold:
                self._speech_started = True
                self._silence_count = 0
                self._total_vad_frames = 0  # reset cap whenever speech is active
            elif self._speech_started:
                self._silence_count += 1

            if self._speech_started and self._silence_count >= self._silence_frames_threshold:
                self._finalize_recording(reason='silence')
                return

        if self._total_vad_frames >= self._max_vad_frames:
            self._finalize_recording(reason='timeout')

    def _finalize_recording(self, reason: str) -> None:
        audio_data = np.concatenate(self._rec_buf)
        self._state = _State.LISTENING

        if reason == 'timeout':
            self.get_logger().warn('Max utterance duration reached')

        # Write WAV in a daemon thread to keep the audio callback non-blocking
        threading.Thread(
            target=self._save_wav,
            args=(audio_data,),
            daemon=True,
        ).start()

    def _save_wav(self, audio_data: np.ndarray) -> None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self._output_dir, f'utterance_{ts}.wav')

        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_data.tobytes())

        self.get_logger().info(f'Utterance saved: {filepath}')
        self._file_pub.publish(String(data=filepath))

    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._stream.stop()
        self._stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WakeWordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
