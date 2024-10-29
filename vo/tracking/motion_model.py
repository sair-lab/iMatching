from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R

class MotionModel:
    class State(Enum):
        UNINITIALIZED = -1
        INITIALIZED = 0
        READY = 1

    def __init__(self) -> None:
        self.velocity = np.zeros(3)
        self.angular_vel = np.zeros(3)
        self.last_R = R.identity()
        self.last_t = np.zeros(3)
        self.last_time = 0.
        self.state = self.State.UNINITIALIZED

    @property
    def ready(self):
        return self.state is self.State.READY

    def predict(self, timestamp: float):
        if not self.ready:
            return None

        dt = timestamp - self.last_time
        rel_R = R.from_rotvec(self.angular_vel * dt)
        rel_t = self.velocity * dt

        new_pose = np.eye(4)
        new_pose[0:3, 0:3] = (rel_R * self.last_R).as_matrix()
        new_pose[0:3, 3] = rel_R.apply(self.last_t) + rel_t
        return new_pose

    def update(self, new_pose: np.ndarray, timestamp: float):
        new_R, new_t = R.from_matrix(new_pose[0:3, 0:3]), new_pose[0:3, 3]
        if self.state in (self.State.INITIALIZED, self.State.READY):
            dt = timestamp - self.last_time
            assert dt != 0
            self.velocity = (new_t - self.last_t) / dt
            self.angular_vel = (new_R * self.last_R.inv()).as_rotvec() / dt
            self.state = self.State.READY
        else:
            self.state = self.State.INITIALIZED

        self.last_time = timestamp
        self.last_R, self.last_t = new_R, new_t
