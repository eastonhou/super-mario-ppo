import cv2
import numpy as np
from common import gcutils

class MetricLogger:
    def __init__(self, path):
        self.path = path

    def train_step(self, loss):
        self.ep_losses.append(loss)

    def eval_step(self, score):
        self.scores.append(score)

    def new_epoch(self):
        self.scores = []
        self.ep_losses = []

    def end_epoch(self, epoch):
        message = f'[epoch {epoch}] loss={np.mean(self.ep_losses):>.2F} scores={self.scores[-1]:>.2F} step={len(self.scores)}'
        lines = list(gcutils.read_all_lines(self.path))
        lines.append(message)
        gcutils.write_all_lines(self.path, lines)

    @property
    def score(self): return self.scores[-1]

class Monitor:
    def __init__(self, width, height, saved_path):
        self.video = cv2.VideoWriter(str(saved_path), 0, 24, (width, height))

    def record(self, image_array):
        self.video.write(image_array[..., ::-1])

def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))
