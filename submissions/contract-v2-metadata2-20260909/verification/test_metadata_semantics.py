"""桥名与位置的业务语义回归，使用合成文件，不访问比赛样本。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import metadata_semantics as metadata


class MetadataTests(unittest.TestCase):
    def test_v2_near_far_missing_and_side(self):
        name, source, distance = metadata.nearest_bridge_name((31.5087, 109.6163), '左幅_第1跨-跨中梁底.JPG')
        self.assertEqual((name, source), ('青树湾1号大桥（左幅）', '公开坐标近邻'))
        self.assertLess(distance, 100)
        name, source, distance = metadata.nearest_bridge_name((31.6250, 109.6350), '右幅_第2跨-1号墩支座.JPG')
        self.assertEqual((name, source), ('未知桥梁（右幅）', '超出公开坐标阈值'))
        self.assertGreater(distance, 2000)
        self.assertEqual(metadata.nearest_bridge_name(None, '左右幅_第3跨.JPG'), ('未知桥梁', '无EXIF坐标', None))

    def test_two_kilometre_boundary_is_inclusive(self):
        for distance, expected in [(2000.0, '公开坐标近邻'), (2000.01, '超出公开坐标阈值')]:
            with patch.object(metadata, '_distance_metres', return_value=distance):
                self.assertEqual(metadata.nearest_bridge_name((31.5, 109.6), 'x.jpg')[1], expected)

    def test_real_exif_read_and_missing_metadata(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '桥梁.jpg'
            exif = Image.Exif()
            exif[34853] = {1: 'N', 2: (31.0, 30.0, 31.32), 3: 'E', 4: (109.0, 36.0, 58.68)}
            Image.new('RGB', (4, 4)).save(path, exif=exif)
            gps = metadata.image_gps(path)
            self.assertAlmostEqual(gps[0], 31.5087, places=5)
            self.assertAlmostEqual(gps[1], 109.6163, places=5)
            self.assertEqual(metadata.bridge_metadata('桥梁', path)['value'], '青树湾1号大桥')
            Image.new('RGB', (4, 4)).save(path)
            self.assertEqual(metadata.bridge_metadata('桥梁', path)['value'], '未知桥梁')
            path.write_bytes(b'broken image')
            self.assertIsNone(metadata.image_gps(path))

    def test_rail_does_not_read_bridge_gps(self):
        with patch.object(metadata, 'image_gps', side_effect=AssertionError('轨道不应匹配桥名')):
            self.assertEqual(metadata.bridge_metadata('轨道', Path('轨道/a.jpg'))['value'], '不适用')

    def test_component_name_and_numeric_identifier_keep_different_meaning(self):
        found = metadata.structure_location('桥梁', '左幅_第3跨-跨中梁底.JPG', '完好', True)
        self.assertEqual(found['value'], '第3跨-跨中梁底')
        self.assertEqual(found['certainty'], 'filename_metadata')
        found = metadata.structure_location('桥梁', '右幅_3_2_1.jpg', '裂缝', False)
        self.assertEqual(found['value'], '3-2-1（文件名编号，构件含义未核验）')
        self.assertEqual(found['certainty'], 'identifier_only')

    def test_camera_sequence_and_arbitrary_filename_are_not_components(self):
        for filename in ['DJI_20260422_001_V.JPG', 'IMG_003.jpg', '20260422.jpg', 'abcdef123.jpg', 'random.jpg']:
            with self.subTest(filename=filename):
                self.assertIsNone(metadata.filename_location(filename))
                location = metadata.structure_location('桥梁', filename, '完好', True)
                self.assertIn('具体构件位置未知', location['value'])
                self.assertNotIn('全桥', location['value'])

    def test_component_filename_with_space_and_photo_counter_is_preserved(self):
        found = metadata.filename_location('左幅_第3跨-跨中梁底 (2).JPG')
        self.assertEqual(found['value'], '第3跨-跨中梁底 (2)')
        self.assertEqual(found['certainty'], 'filename_metadata')

    def test_type_inference_is_not_presented_as_observed_component(self):
        location = metadata.structure_location('轨道', 'opaque.jpg', '破损、支座锈蚀', False)
        self.assertEqual(location['value'], '支座、结构表面（按病害类型推断，具体位置未知）')
        self.assertEqual(location['certainty'], 'type_inference')
        # 单独“裂缝”不支持混凝土材质。
        self.assertNotIn('混凝土', metadata.structure_location('桥梁', 'a.jpg', '裂缝', False)['value'])

    def test_domain_names_image_coordinates_and_full_bridge_are_rejected(self):
        location = metadata.structure_location('桥梁', 'a.jpg', '裂缝', False)
        for name in ['桥梁', '轨道']:
            with self.assertRaisesRegex(ValueError, '分类目录'):
                metadata.validate_metadata('桥梁', {'value': name}, location)
        bridge = {'value': '未知桥梁', 'source': '无EXIF坐标', 'distance_metres': None}
        for position in ['图像中心区域（候选定位）', '全桥', '桥梁结构', '']:
            with self.assertRaisesRegex(ValueError, '结构位置'):
                metadata.validate_metadata('桥梁', bridge, {'value': position})


if __name__ == '__main__':
    unittest.main()
