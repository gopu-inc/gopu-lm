import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gopu_lm import Model, _write_safetensors


class FakeLibrary:
    def gopu_lm_save(self, _handle, path):
        metadata = {
            "arch": "gopu-lm",
            "vocab": "270",
            "d": "8",
            "layers": "1",
            "heads": "2",
            "ctx": "16",
            "ff": "16",
            "future": "0",
            "n_specials": "14",
            "tok_merges": "",
        }
        tensors = {
            "tok_emb": ("F32", [150], b"\0" * 600),
            "pos_emb": ("F32", [150], b"\0" * 600),
        }
        _write_safetensors(os.fsdecode(path), metadata, tensors)
        return 0


class SaveModelExportTests(unittest.TestCase):
    def test_save_model_exports_huggingface_folder_with_bounded_shards(self):
        model = Model.__new__(Model)
        model._h = 1
        model._L = FakeLibrary()
        with tempfile.TemporaryDirectory() as temp_dir, patch("gopu_lm._HF_MAX_FILE_SIZE", 900):
            output = os.path.join(temp_dir, "my-model")
            model.save_model(output)

            expected = {
                "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "vocab.txt", "chat_template.jinja", "configuration_gopu.py", "modeling_gopu.py",
                "model.safetensors.index.json",
            }
            self.assertTrue(expected.issubset(set(os.listdir(output))))
            with open(os.path.join(output, "model.safetensors.index.json"), encoding="utf-8") as f:
                index = json.load(f)
            self.assertEqual(set(index["weight_map"]), {"tok_emb", "pos_emb"})
            for shard in set(index["weight_map"].values()):
                self.assertLessEqual(os.path.getsize(os.path.join(output, shard)), 900)
            with open(os.path.join(output, "config.json"), encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["model_type"], "gopu_lm")
            with open(os.path.join(output, "tokenizer.json"), encoding="utf-8") as f:
                tokenizer = json.load(f)
            self.assertEqual(tokenizer["model"]["type"], "BPE")

    def test_save_model_rejects_a_tensor_that_cannot_fit_a_shard(self):
        model = Model.__new__(Model)
        model._h = 1
        model._L = FakeLibrary()
        with tempfile.TemporaryDirectory() as temp_dir, patch("gopu_lm._HF_MAX_FILE_SIZE", 300):
            with self.assertRaisesRegex(ValueError, "25 000 000 octets"):
                model.save_model(os.path.join(temp_dir, "too-large"))


if __name__ == "__main__":
    unittest.main()
