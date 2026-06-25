import unittest

import torch
import torch.nn as nn

from eval.baseline.apet import ApETPruner
from eval.baseline.divprune import DivPrunePruner
from eval.baseline.fastv_mesh import FastVMeshPruner
from eval.baseline.otprune import OTPrunePruner
from eval.baseline.tome import ToMePruner
from eval.eva01_backend import (
    build_eva01_inputs_embeds_from_mesh_tokens,
    build_eva01_prompt_inputs,
    generate_eva01_caption_from_mesh_tokens,
    map_vq_indices_to_eva_patches,
    select_eva01_mesh_tokens,
    vqvae_latent_centers,
)
from eval.pruners.baseline import NoPruningPruner, RandomPruningPruner, UniformDownsamplingPruner


class FakeTokenizer:
    mesh_token = "<|mesh_und_pad|>"
    mesh_token_id = 7
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(str(m["content"]) for m in messages)
        if add_generation_prompt:
            rendered = f"{rendered}\n<assistant>"
        return rendered

    def __call__(self, text, add_special_tokens=False, return_tensors="pt", **kwargs):
        ids = []
        for part in str(text).replace("\n", " ").split():
            if part == self.mesh_token:
                ids.append(self.mesh_token_id)
            else:
                ids.append(100 + (len(ids) % 50))
        if not ids:
            ids = [101]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def convert_tokens_to_ids(self, token):
        if token == self.mesh_token:
            return self.mesh_token_id
        return 1

    def batch_decode(self, *args, **kwargs):
        return ["decoded caption"]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.mesh_und_token = FakeTokenizer.mesh_token
        self.mesh_und_token_id = FakeTokenizer.mesh_token_id

    def apply_chat_template(self, *args, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "mesh_und_values": torch.zeros(1, 8, 6, dtype=torch.float32),
        }

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


class FakeQwen:
    def __init__(self):
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        tail = torch.tensor([[11, 12]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, tail], dim=1)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 4)
        self.mesh_und_connector = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.emb.weight.zero_()
            self.mesh_und_connector.weight.copy_(torch.eye(4))
        self.qwen3vl = FakeQwen()
        self.config = type("Config", (), {"mesh_und_token_id": FakeTokenizer.mesh_token_id})()
        self._output_dtype = torch.float32

    @property
    def device(self):
        return torch.device("cpu")

    def get_input_embeddings(self):
        return self.emb


class EVA01AdapterTests(unittest.TestCase):
    def test_variable_prompt_and_embedding_replacement(self):
        processor = FakeProcessor()
        model = FakeModel()
        mesh_tokens = torch.arange(16, dtype=torch.float32).view(4, 4)

        inputs = build_eva01_prompt_inputs(processor, "describe it", 4, device=torch.device("cpu"))
        input_ids = inputs["input_ids"]
        self.assertEqual(4, int(input_ids.eq(FakeTokenizer.mesh_token_id).sum().item()))

        embeds = build_eva01_inputs_embeds_from_mesh_tokens(
            model,
            processor,
            input_ids,
            mesh_tokens,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(embeds[input_ids.eq(FakeTokenizer.mesh_token_id)], mesh_tokens))

    def test_select_mesh_tokens_preserves_cls_and_patch_order(self):
        mesh_tokens = torch.arange(18, dtype=torch.float32).view(6, 3)
        selected = select_eva01_mesh_tokens(mesh_tokens, [2, 4])
        self.assertTrue(torch.equal(selected[0], mesh_tokens[0]))
        self.assertTrue(torch.equal(selected[1], mesh_tokens[3]))
        self.assertTrue(torch.equal(selected[2], mesh_tokens[5]))

    def test_generate_from_pruned_tokens_reports_counts(self):
        processor = FakeProcessor()
        model = FakeModel()
        mesh_tokens = torch.zeros(3, 4)
        text, _elapsed, n_in, n_out = generate_eva01_caption_from_mesh_tokens(
            model,
            processor,
            mesh_tokens,
            "caption",
            device=torch.device("cpu"),
        )
        self.assertEqual("decoded caption", text)
        self.assertEqual(2, n_out)
        self.assertEqual(model.qwen3vl.last_kwargs["input_ids"].shape[1], n_in)
        self.assertEqual(3, int(model.qwen3vl.last_kwargs["input_ids"].eq(FakeTokenizer.mesh_token_id).sum().item()))

    def test_baseline_pruners_accept_512_and_1024_tokens(self):
        pruner_classes = [
            NoPruningPruner,
            RandomPruningPruner,
            UniformDownsamplingPruner,
            DivPrunePruner,
            ApETPruner,
            OTPrunePruner,
            ToMePruner,
            FastVMeshPruner,
        ]
        for n in (512, 1024):
            features = torch.randn(n, 8)
            embed = nn.Embedding.from_pretrained(features, freeze=True)
            token_ids = torch.arange(n, dtype=torch.long)
            for cls in pruner_classes:
                kwargs = {"basis_token_num": 8} if cls is ApETPruner else {}
                pruner = cls(keep_ratio=0.25, seed=3, **kwargs)
                pruned, _meta = pruner.prune(token_ids, None, vq_embeddings=embed)
                expected = n if cls is NoPruningPruner else max(1, int(round(n * 0.25)))
                self.assertEqual(expected, int(pruned.numel()), f"{cls.__name__} n={n}")
                self.assertGreaterEqual(int(pruned.min().item()), 0)
                self.assertLess(int(pruned.max().item()), n)

    def test_vq_to_eva_mapping_fills_duplicate_nearest_patches(self):
        base = vqvae_latent_centers()[0]
        centers = torch.stack(
            [
                base,
                base + torch.tensor([2.0, 0.0, 0.0]),
                base + torch.tensor([0.0, 2.0, 0.0]),
                base + torch.tensor([0.0, 0.0, 2.0]),
            ],
            dim=0,
        )
        first, diag = map_vq_indices_to_eva_patches([0, 0, 0], centers, target_count=3)
        second, _ = map_vq_indices_to_eva_patches([0, 0, 0], centers, target_count=3)
        self.assertEqual(first, second)
        self.assertEqual(3, len(first))
        self.assertEqual(2, diag["duplicate_mapped_count"])
        self.assertEqual(2, diag["filled_count"])


if __name__ == "__main__":
    unittest.main()
