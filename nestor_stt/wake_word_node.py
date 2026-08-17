import numpy as np
import sounddevice as sd
from openwakeword.model import Model
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WakeWordNode(Node):
    def __init__(self):
        super().__init__('wake_word_node')

        self.declare_parameter('model_paths', [''])
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('chunk_size', 1280)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('cooldown_s', 1.0)

        model_paths = self.get_parameter('model_paths').get_parameter_value().string_array_value
        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value
        self.chunk_size = self.get_parameter('chunk_size').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        cooldown_s = self.get_parameter('cooldown_s').get_parameter_value().double_value

        valid_paths = [p for p in model_paths if p]
        if valid_paths:
            self.get_logger().info(f'Loading custom models: {valid_paths}')
            self.oww_model = Model(wakeword_model_paths=valid_paths)
        else:
            self.get_logger().info('Loading default pre-trained models')
            self.oww_model = Model()

        self._cooldown_ticks = int(cooldown_s * self.sample_rate / self.chunk_size)
        self._cooldown_remaining: dict[str, int] = {}

        self._pub = self.create_publisher(String, 'stt/wake_word_detected', 10)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self.get_logger().info('Wake word node ready, listening for wake words')

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            self.get_logger().warn(f'Audio stream status: {status}')

        audio = indata[:, 0]  # mono
        predictions = self.oww_model.predict(audio)

        for word, score in predictions.items():
            remaining = self._cooldown_remaining.get(word, 0)
            if remaining > 0:
                self._cooldown_remaining[word] = remaining - 1
                continue

            if score >= self.threshold:
                self.get_logger().info(f'Wake word detected: "{word}" (score={score:.3f})')
                self._pub.publish(String(data=word))
                self._cooldown_remaining[word] = self._cooldown_ticks

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
