import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn_rwkv_story_gen import SocialNarrativeModel
from semantic_memory import build_memory_from_text


def test_routes_geography_query_to_delhi_cluster():
    graph = build_memory_from_text(
        "Delhi is a city in India. India is a country. "
        "Java is a programming language. Java is used for software development."
    )

    route = graph.route_query("delhi kahan h")
    facts = "\n".join(route["facts"])
    clusters = " ".join(cluster["label"].lower() for cluster in route["clusters"])

    assert "delhi --is_a--> city" in facts
    assert "delhi --located_in--> india" in facts
    assert "geography" in clusters


def test_routes_misspelled_java_query_to_programming_cluster():
    graph = build_memory_from_text(
        "Delhi is a city in India. India is a country. "
        "Java is a programming language. Java is used for software development."
    )

    route = graph.route_query("jvaa code likho")
    facts = "\n".join(route["facts"])
    clusters = " ".join(cluster["label"].lower() for cluster in route["clusters"])

    assert "java" in facts
    assert "programming" in clusters


def test_model_uses_active_node_mask_for_lif_routing():
    model = SocialNarrativeModel(vocab_size=32, num_nodes=8, dim=16, num_layers=2, max_seq=16)
    h = [torch.zeros((8, 16)) for _ in range(2)]
    active_mask = torch.tensor([True, True, False, False, False, False, False, False])

    logits, new_h = model(
        torch.tensor([1, 2, 3]),
        torch.arange(8),
        {"active_node_mask": active_mask},
        h,
    )

    assert logits.shape == (3, 32)
    assert len(new_h) == 2
    assert model.last_active_nodes
    assert set(model.last_active_nodes[-1]).issubset({0, 1})
