"""
Unit tests for model architecture constraints and forward passes.
"""

import unittest
import torch
from src.model import get_model, set_backbone_trainable


class TestModels(unittest.TestCase):

    def test_forbidden_mobilenet(self):
        with self.assertRaises(AssertionError):
            get_model("mobilenet_v2", num_classes=29, pretrained=False)

    def test_shufflenet_v2_shape(self):
        m = get_model("shufflenet_v2", num_classes=29, pretrained=False)
        x = torch.randn(2, 3, 112, 112)
        out = m(x)
        self.assertEqual(out.shape, (2, 29))

    def test_resnet18_shape(self):
        m = get_model("resnet18", num_classes=29, pretrained=False)
        x = torch.randn(2, 3, 112, 112)
        out = m(x)
        self.assertEqual(out.shape, (2, 29))

    def test_regularized_head_shape_and_metadata(self):
        m = get_model(
            "shufflenet_v2",
            num_classes=29,
            pretrained=False,
            head_type="regularized",
            hidden_dim=128,
        )
        out = m(torch.randn(2, 3, 112, 112))
        self.assertEqual(out.shape, (2, 29))
        self.assertEqual(m.head_type, "regularized")
        self.assertEqual(m.hidden_dim, 128)

    def test_linear_head_shape(self):
        m = get_model(
            "resnet18",
            num_classes=29,
            pretrained=False,
            head_type="linear",
        )
        out = m(torch.randn(2, 3, 112, 112))
        self.assertEqual(out.shape, (2, 29))

    def test_invalid_head_type_is_rejected(self):
        with self.assertRaises(ValueError):
            get_model(
                "shufflenet_v2",
                num_classes=29,
                pretrained=False,
                head_type="unknown",
            )

    def test_freeze_backbone(self):
        m = get_model("shufflenet_v2", num_classes=29, pretrained=False)
        set_backbone_trainable(m, trainable=False)
        for name, param in m.named_parameters():
            if "fc" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
