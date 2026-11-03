import unittest
import numpy as np
from origami_bench.diagram.layer_controller import motion


class PacketFoldFrames(unittest.TestCase):
    """A 180 deg fold that carries a >=2-layer stack uses the finer angular step.

    The single-layer schedule (0.5 deg/frame) under-samples the outer layer's
    transient crease angle, which the band-memory grader then scores as a missed
    fold (seen on O03). The packet schedule reproduces the published O03
    reference evidence: 464 frames for 180 deg.
    """

    def setUp(self):
        # 4x4 grid on a 0.15 m sheet, so the moving row is 0.075 m from the axis
        g = np.linspace(0.0, 0.15, 4)
        self.v = np.array([[x, y, 0.0] for y in g for x in g])
        self.ids = list(range(4))
        self.center = np.array([0.0, 0.075, 0.0])
        self.axis = np.array([1.0, 0.0, 0.0])

    def test_single_layer_schedule_is_unchanged(self):
        self.assertEqual(motion(self.ids, self.center, self.axis, 180.0, self.v)['frames'], 360)

    def test_packet_schedule_matches_reference_evidence(self):
        self.assertEqual(motion(self.ids, self.center, self.axis, 180.0, self.v, packet=True)['frames'], 464)

    def test_packet_schedule_scales_with_angle(self):
        self.assertEqual(motion(self.ids, self.center, self.axis, 90.0, self.v, packet=True)['frames'], 232)


if __name__ == '__main__':
    unittest.main()
