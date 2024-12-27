import unittest
import os
from paligemma_multitask.object_detection import detect_objects

class TestObjectDetection(unittest.TestCase):
    def setUp(self):
        # 使用正确的数据集路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_path = os.path.join(base_dir, "dataset")
        self.annotation_path = os.path.join(base_dir, "dataset", "_annotations.train1.jsonl")

    def test_detect_objects(self):
        # 确保路径存在
        self.assertTrue(os.path.exists(self.image_path), f"Image path does not exist: {self.image_path}")
        self.assertTrue(os.path.exists(self.annotation_path), f"Annotation path does not exist: {self.annotation_path}")
        
        # 运行检测
        detections = detect_objects(
            image_path=self.image_path,
            annotation_path=self.annotation_path,
            num_images=2  # 测试时使用较少的图像
        )
        
        # 验证检测结果
        self.assertIsNotNone(detections)
        self.assertTrue(hasattr(detections, 'xyxy'))

if __name__ == '__main__':
    unittest.main() 