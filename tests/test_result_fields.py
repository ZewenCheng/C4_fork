from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from result_fields import defect_description, defect_location, filename_side, nearest_bridge_name, rating_scale


class ResultFieldRulesTest(unittest.TestCase):
    def test_bridge_near_public_reference_and_explicit_side(self):
        name, source, distance = nearest_bridge_name((31.5087, 109.6163), "左幅_第1跨-跨中梁底.JPG")
        self.assertEqual(name, "青树湾1号大桥（左幅）")
        self.assertEqual(source, "公开坐标近邻")
        self.assertIsNotNone(distance)
        self.assertLess(distance, 100)

    def test_bridge_far_from_reference_is_not_fabricated(self):
        name, source, distance = nearest_bridge_name((31.6250, 109.6350), "右幅_第2跨-1号墩支座.JPG")
        self.assertEqual(name, "未知桥梁（右幅）")
        self.assertEqual(source, "超出公开坐标阈值")
        self.assertIsNotNone(distance)
        self.assertGreater(distance, 2_000)

    def test_filename_and_label_rules(self):
        self.assertIsNone(filename_side("左右幅_第3跨-跨中梁底.JPG"))
        self.assertEqual(defect_location("桥梁", "左幅_第3跨-跨中梁底.JPG", "完好"), "第3跨-跨中梁底")
        self.assertEqual(defect_location("桥梁", "DJI_20260422_001_V.JPG", "完好"), "全桥")
        self.assertEqual(defect_location("轨道", "x.jpg", "破损、支座锈蚀"), "支座、结构表面")
        self.assertEqual(defect_description("裂缝(混凝土裂缝)"), "混凝土构件可见裂缝")
        self.assertEqual(rating_scale("完好"), "1")
        self.assertEqual(rating_scale("破损"), "2")
        self.assertEqual(rating_scale("钢筋锈蚀、破损"), "3")


if __name__ == "__main__":
    unittest.main()
