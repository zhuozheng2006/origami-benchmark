"""Static rendering checks only; no physical simulation or reference access."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from origami_bench.diagram.depth_renderer_v15 import view_projection, raster_v15, board_ticks, render_public_v15


class PaperViewportTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[-.075, -.15, 0], [.075, -.15, 0], [.075, .15, 0], [-.075, .15, 0]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])

    def assert_complete(self, vertices):
        for view in ['top', 'oblique', 'bottom_oblique']:
            with self.subTest(view=view):
                projection = view_projection(vertices, view, 128)
                # Every original vertex projects well inside the image; the
                # convexity of triangles then guarantees no face gets clipped.
                self.assertGreater(float(projection['screen'].min()), 4)
                self.assertLess(float(projection['screen'].max()), 123)
                _, owner = raster_v15(vertices, self.faces, 128, view)
                self.assertTrue(np.any(owner >= 0))
                self.assertTrue(np.all(owner[[0, -1], :] == -1))
                self.assertTrue(np.all(owner[:, [0, -1]] == -1))

    def test_non_square_sheet_all_three_views_complete(self):
        self.assert_complete(self.vertices)

    def test_offset_sheet_all_three_views_complete(self):
        self.assert_complete(self.vertices + [.14, -.12, .06])

    def test_rotated_tilted_sheet_all_three_views_complete(self):
        theta = .6
        rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        self.assert_complete(self.vertices @ rotation + [.05, .03, .1])

    def test_square_default_view_and_world_origin_are_preserved(self):
        square = self.vertices.copy(); square[:, 1] *= .5
        projection = view_projection(square, 'top', 512)
        self.assertEqual(projection['half_extent_m'], .12)
        self.assertTrue(projection['world_origin_fixed'])
        origin = view_projection(np.vstack([square, [0, 0, 0]]), 'top', 512)['screen'][-1]
        np.testing.assert_allclose(origin, [255.5, 255.5])

    def test_rectangle_ticks_cover_real_board_edges_without_normalization(self):
        limit = view_projection(self.vertices, 'top')['half_extent_m']
        ticks, step = board_ticks(limit)
        self.assertIn(-.5, ticks); self.assertIn(1.5, ticks)
        self.assertEqual(step, .25)
        # The rectangle remains .15 x .30 metres / 1 x 2 board units.
        self.assertGreater(limit, .15)
        p = view_projection(self.vertices, 'top')['screen']
        self.assertAlmostEqual((p[2, 1] - p[1, 1]) / (p[1, 0] - p[0, 0]), -2.)

    def test_rendered_canvas_is_complete_and_has_no_private_input(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'paper.png'
            render_public_v15(dict(vertices=self.vertices.tolist(), faces=self.faces.tolist()), target)
            with Image.open(target) as image:
                self.assertEqual(image.size, (1680, 630))

    def test_invalid_vertices_fail_explicitly(self):
        for vertices in [[], [[0, 0, float('nan')]], [[0, 0]]]:
            with self.assertRaises(ValueError): view_projection(vertices)


if __name__ == '__main__': unittest.main()
