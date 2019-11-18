from sleap.nn import tracking

from collections import deque
import numpy as np
import tensorflow as tf


class TestTracking(tf.test.TestCase):
    def setUp(self):
        from sleap.skeleton import Skeleton
        from sleap.instance import Instance, Point

        self.skeleton = Skeleton("test skeleton")
        self.skeleton.add_nodes(["a", "b", "c", "d"])
        self.skeleton.add_edge("a", "b")
        self.skeleton.add_edge("a", "c")
        self.skeleton.add_edge("b", "d")
        self.skeleton.add_edge("c", "d")

        inst_a = Instance(skeleton=self.skeleton)
        inst_a["a"] = Point(0, 0)
        inst_a["b"] = Point(10, 0)
        inst_a["c"] = Point(0, 10)
        inst_a["d"] = Point(10, 10)

        # just first two points from inst a
        inst_a_piece = Instance(skeleton=self.skeleton)
        inst_a_piece["a"] = Point(0, 0)
        inst_a_piece["b"] = Point(10, 0)

        # different nodes at same points as a
        inst_rot_a = Instance(skeleton=self.skeleton)
        inst_rot_a["a"] = Point(10, 10)
        inst_rot_a["b"] = Point(0, 0)
        inst_rot_a["c"] = Point(10, 0)
        inst_rot_a["d"] = Point(0, 10)

        # same shape as a but shifted (100, 100)
        inst_b = Instance(skeleton=self.skeleton)
        inst_b["a"] = Point(100, 100)
        inst_b["b"] = Point(110, 100)
        inst_b["c"] = Point(100, 110)
        inst_b["d"] = Point(110, 110)

        # slight wiggle from a
        inst_sim_a = Instance(skeleton=self.skeleton)
        inst_sim_a["a"] = Point(2, 3)
        inst_sim_a["b"] = Point(9, 1)
        inst_sim_a["c"] = Point(0, 12)
        inst_sim_a["d"] = Point(10, 10)

        # same center as a, just bigger
        inst_cent_a = Instance(skeleton=self.skeleton)
        inst_cent_a["a"] = Point(-10, -10)
        inst_cent_a["b"] = Point(20, -10)
        inst_cent_a["c"] = Point(-10, 20)
        inst_cent_a["d"] = Point(20, 20)

        self.inst_a = inst_a
        self.inst_b = inst_b
        self.inst_a_piece = inst_a_piece
        self.inst_sim_a = inst_sim_a
        self.inst_rot_a = inst_rot_a
        self.inst_cent_a = inst_cent_a

    def test_instance_similarity(self):
        self.assertEqual(
            tracking.instance_similarity(self.inst_a, self.inst_b),
            tracking.instance_similarity(self.inst_b, self.inst_a),
        )

        self.assertLess(
            tracking.instance_similarity(self.inst_a, self.inst_b),
            tracking.instance_similarity(self.inst_a, self.inst_sim_a),
        )

        self.assertLess(
            tracking.instance_similarity(self.inst_a, self.inst_cent_a),
            tracking.instance_similarity(self.inst_a, self.inst_sim_a),
        )

    def test_centroid_distance(self):
        self.assertEqual(
            tracking.centroid_distance(self.inst_a, self.inst_b),
            tracking.centroid_distance(self.inst_b, self.inst_a),
        )

        self.assertLess(
            tracking.centroid_distance(self.inst_a, self.inst_b),
            tracking.centroid_distance(self.inst_a, self.inst_sim_a),
        )

        self.assertLess(
            tracking.centroid_distance(self.inst_a, self.inst_b),
            tracking.centroid_distance(self.inst_a, self.inst_cent_a),
        )

        self.assertLess(
            tracking.centroid_distance(self.inst_a, self.inst_sim_a),
            tracking.centroid_distance(self.inst_a, self.inst_cent_a),
        )

        self.assertEqual(tracking.centroid_distance(self.inst_a, self.inst_cent_a), 0)

        self.assertEqual(tracking.centroid_distance(self.inst_a, self.inst_rot_a), 0)

    def test_instance_iou(self):
        self.assertEqual(
            tracking.instance_iou(self.inst_a, self.inst_b),
            tracking.instance_iou(self.inst_b, self.inst_a),
        )

        self.assertEqual(tracking.instance_iou(self.inst_a, self.inst_b), 0)

        self.assertEqual(
            tracking.instance_iou(self.inst_a, self.inst_a),
            tracking.instance_iou(self.inst_a, self.inst_rot_a),
        )

        self.assertLess(
            tracking.instance_iou(self.inst_a, self.inst_b),
            tracking.instance_iou(self.inst_a, self.inst_sim_a),
        )

    def test_hungarian_matching(self):
        cost_matrix = np.array([[0, 1, 2], [1, 10, 15],])
        matches = tracking.hungarian_matching(cost_matrix)
        self.assertEqual(matches, [(0, 1), (1, 0)])

        cost_matrix = np.array([[-5, np.inf], [np.inf, -5],])
        matches = tracking.hungarian_matching(cost_matrix)
        self.assertEqual(matches, [(0, 0), (1, 1)])

    def test_greedy_matching(self):
        cost_matrix = np.array([[0, 1, 2], [1, 10, 15],])
        matches = tracking.greedy_matching(cost_matrix)
        self.assertEqual(matches, [(0, 0), (1, 1)])

    def test_FlowCandidateMaker(self):

        from sleap.io.dataset import Labels

        labels = Labels.load_file("tests/data/json_format_v1/centered_pair.json")

        match_queue = deque(maxlen=3)
        frame_0_instances = labels.find_first(labels.videos[0], 0).instances

        match_queue.append(
            tracking.MatchedInstance(0, frame_0_instances, labels.videos[0][0])
        )

        candidates = tracking.FlowCandidateMaker().get_candidates(
            match_queue, 1, labels.videos[0][1]
        )

        self.assertLen(frame_0_instances, 2)
        self.assertLen(candidates, 2)

        # forelegR3 [node 11] should shift down on the first instance
        self.assertLess(
            frame_0_instances[0].points_array[11, 1], candidates[0].points_array[11, 1]
        )

        # forelegR3 [node 11] should shift up on the second instance
        self.assertGreater(
            frame_0_instances[1].points_array[11, 1], candidates[1].points_array[11, 1]
        )

        # hindlegR3 [node 23] should shift left on both instances
        self.assertLess(
            frame_0_instances[0].points_array[23, 0], candidates[0].points_array[23, 0]
        )

        self.assertLess(
            frame_0_instances[1].points_array[23, 0], candidates[1].points_array[23, 0],
        )

    def test_SimpleCandidateMaker(self):

        match_queue = deque(maxlen=3)
        match_queue.append(tracking.MatchedInstance(0, [self.inst_a, self.inst_b]))
        match_queue.append(tracking.MatchedInstance(1, [self.inst_sim_a]))
        match_queue.append(tracking.MatchedInstance(2, [self.inst_a_piece]))

        candidates = tracking.SimpleCandidateMaker().get_candidates(match_queue)

        self.assertLen(candidates, 4)
        self.assertContainsSubset(
            [self.inst_a, self.inst_b, self.inst_sim_a, self.inst_a_piece], candidates
        )

        candidates = tracking.SimpleCandidateMaker(min_points=3).get_candidates(
            match_queue
        )

        self.assertLen(candidates, 3)
        self.assertContainsSubset(
            [self.inst_a, self.inst_b, self.inst_sim_a], candidates
        )
