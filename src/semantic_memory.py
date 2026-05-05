import json
import re
from collections import Counter, defaultdict, deque
from difflib import get_close_matches
from pathlib import Path

DEFAULT_MEMORY_PATH = "memory_graph.json"
DEFAULT_TOP_K_CLUSTERS = 3
DEFAULT_MAX_CONTEXT_FACTS = 20
DEFAULT_GRAPH_HOPS = 2

RELATIONS = {
    "is_a",
    "part_of",
    "located_in",
    "used_for",
    "related_to",
    "example_of",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
    "write",
    "program",
    "code",
    "lines",
    "line",
    "me",
    "mai",
    "kahan",
    "kaha",
    "hai",
    "h",
}

DOMAIN_SEEDS = {
    "geography": {
        "city",
        "country",
        "capital",
        "state",
        "province",
        "river",
        "location",
        "located",
        "india",
        "delhi",
    },
    "programming": {
        "programming",
        "computer",
        "software",
        "code",
        "class",
        "function",
        "method",
        "compiler",
        "java",
        "python",
        "javascript",
    },
    "ai": {
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "neural",
        "model",
        "reasoning",
        "language",
    },
}


def normalize_label(text):
    text = re.sub(r"[_\-]+", " ", str(text).lower())
    text = re.sub(r"[^a-z0-9\s+#.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [
        token
        for token in re.findall(r"[a-z0-9+#.]+", normalize_label(text))
        if token and token not in STOPWORDS and len(token) > 1
    ]


def node_id(label):
    normalized = normalize_label(label)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "node"


class SemanticMemoryGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.communities = {}
        self.node_community = {}
        self.cluster_labels = {}

    def add_node(self, label, kind="concept", aliases=None):
        label = normalize_label(label)
        if not label:
            return None
        nid = node_id(label)
        existing = self.nodes.get(nid)
        if existing:
            existing["count"] = existing.get("count", 1) + 1
            for alias in aliases or []:
                alias = normalize_label(alias)
                if alias and alias not in existing["aliases"]:
                    existing["aliases"].append(alias)
            return nid
        self.nodes[nid] = {
            "id": nid,
            "label": label,
            "kind": kind,
            "aliases": [normalize_label(a) for a in aliases or [] if normalize_label(a)],
            "count": 1,
        }
        return nid

    def add_edge(self, source, relation, target, evidence=None, confidence="EXTRACTED"):
        if relation not in RELATIONS:
            relation = "related_to"
        src = self.add_node(source)
        dst = self.add_node(target)
        if not src or not dst or src == dst:
            return
        edge = {
            "source": src,
            "target": dst,
            "relation": relation,
            "evidence": evidence or "",
            "confidence": confidence,
        }
        key = (src, dst, relation)
        if key not in {(e["source"], e["target"], e["relation"]) for e in self.edges}:
            self.edges.append(edge)

    def ingest_text(self, text):
        self.add_seed_facts()
        for sentence in split_sentences(text):
            self._extract_sentence(sentence)
        self._add_domain_edges()
        self.cluster()

    def add_seed_facts(self):
        seeds = [
            ("delhi", "is_a", "city"),
            ("delhi", "located_in", "india"),
            ("india", "is_a", "country"),
            ("artificial intelligence", "is_a", "field of computer science"),
            ("artificial intelligence", "used_for", "building machines that can perform intelligent tasks"),
            ("machine learning", "is_a", "branch of artificial intelligence"),
            ("java", "is_a", "programming language"),
            ("java", "used_for", "software development"),
        ]
        for source, relation, target in seeds:
            self.add_edge(source, relation, target, confidence="SEEDED")

    def _extract_sentence(self, sentence):
        clean = normalize_label(sentence)
        if len(clean) < 4:
            return

        patterns = [
            (r"\b([a-z][a-z0-9+#. ]{1,30})\s+is\s+a[n]?\s+(city|country|state|province|capital)\s+in\s+([a-z][a-z0-9+#. ]{1,30})\b", "typed_location"),
            (r"\b([a-z][a-z0-9+#. ]{1,40})\s+is\s+a[n]?\s+([a-z][a-z0-9+#. ]{1,40})\b", "is_a"),
            (r"\b([a-z][a-z0-9+#. ]{1,40})\s+is\s+an?\s+([a-z][a-z0-9+#. ]{1,40})\b", "is_a"),
            (r"\b([a-z][a-z0-9+#. ]{1,40})\s+is\s+the\s+capital\s+of\s+([a-z][a-z0-9+#. ]{1,40})\b", "located_in"),
            (r"\b([a-z][a-z0-9+#. ]{1,40})\s+is\s+located\s+in\s+([a-z][a-z0-9+#. ]{1,40})\b", "located_in"),
            (r"\b([a-z][a-z0-9+#. ]{1,40})\s+is\s+part\s+of\s+([a-z][a-z0-9+#. ]{1,40})\b", "part_of"),
            (r"\b([a-z][a-z0-9+#. ]{1,40})\s+is\s+used\s+for\s+([a-z][a-z0-9+#. ]{1,40})\b", "used_for"),
        ]
        for pattern, relation in patterns:
            match = re.search(pattern, clean)
            if match:
                source = compact_phrase(match.group(1))
                if relation == "typed_location":
                    place_type = compact_phrase(match.group(2))
                    location = compact_phrase(match.group(3))
                    self.add_edge(source, "is_a", place_type, evidence=sentence)
                    self.add_edge(source, "located_in", location, evidence=sentence)
                    continue
                raw_target = match.group(2)
                if relation == "is_a" and re.search(r"\b(city|country|state|province|capital)\s+in\b", raw_target):
                    continue
                target = compact_phrase(raw_target)
                self.add_edge(source, relation, target, evidence=sentence)

        nouns = [token for token in tokenize(clean) if len(token) > 2]
        for token in nouns[:12]:
            self.add_node(token)
        for left, right in zip(nouns[:10], nouns[1:11]):
            self.add_edge(left, "related_to", right, evidence=sentence, confidence="INFERRED")

    def _add_domain_edges(self):
        labels = {nid: set(tokenize(node["label"])) for nid, node in self.nodes.items()}
        for domain, seeds in DOMAIN_SEEDS.items():
            self.add_node(domain, kind="cluster_hint")
            for nid, words in labels.items():
                if words & seeds:
                    self.add_edge(self.nodes[nid]["label"], "related_to", domain, confidence="INFERRED")

    def cluster(self):
        try:
            communities = self._cluster_networkx()
        except Exception:
            communities = self._cluster_fallback()
        self.communities = {str(cid): sorted(nodes) for cid, nodes in communities.items()}
        self.node_community = {
            nid: str(cid)
            for cid, nodes in self.communities.items()
            for nid in nodes
        }
        self.cluster_labels = {
            str(cid): self._label_cluster(nodes)
            for cid, nodes in self.communities.items()
        }

    def _cluster_networkx(self):
        import networkx as nx

        graph = nx.Graph()
        for nid in self.nodes:
            graph.add_node(nid)
        for edge in self.edges:
            graph.add_edge(edge["source"], edge["target"])
        if graph.number_of_edges() == 0:
            return {i: [nid] for i, nid in enumerate(sorted(self.nodes))}
        try:
            from graspologic.partition import leiden

            partition = leiden(graph)
            raw = defaultdict(list)
            for nid, cid in partition.items():
                raw[cid].append(nid)
        except Exception:
            groups = nx.community.louvain_communities(graph, seed=42, threshold=1e-4)
            raw = {cid: list(nodes) for cid, nodes in enumerate(groups)}
        ordered = sorted(raw.values(), key=len, reverse=True)
        return {cid: sorted(nodes) for cid, nodes in enumerate(ordered)}

    def _cluster_fallback(self):
        groups = defaultdict(list)
        for nid, node in self.nodes.items():
            words = set(tokenize(node["label"]))
            domain = "general"
            for name, seeds in DOMAIN_SEEDS.items():
                if words & seeds:
                    domain = name
                    break
            groups[domain].append(nid)
        ordered = sorted(groups.values(), key=len, reverse=True)
        return {cid: sorted(nodes) for cid, nodes in enumerate(ordered)}

    def _label_cluster(self, nodes):
        counts = Counter()
        for nid in nodes:
            counts.update(tokenize(self.nodes[nid]["label"]))
        for domain, seeds in DOMAIN_SEEDS.items():
            if counts and sum(counts[s] for s in seeds) >= 1:
                return domain.title()
        common = [word for word, _ in counts.most_common(3)]
        return " ".join(common).title() if common else "General"

    def route_query(self, query, top_k_clusters=DEFAULT_TOP_K_CLUSTERS, max_facts=DEFAULT_MAX_CONTEXT_FACTS, hops=DEFAULT_GRAPH_HOPS):
        q_tokens = set(tokenize(query))
        matched = self._match_nodes(query, q_tokens)
        cluster_scores = Counter()
        for nid, score in matched.items():
            cid = self.node_community.get(nid)
            if cid is not None:
                cluster_scores[cid] += score
        for cid, label in self.cluster_labels.items():
            if q_tokens & set(tokenize(label)):
                cluster_scores[cid] += 2.0
        selected = [cid for cid, _ in cluster_scores.most_common(top_k_clusters)]
        facts = self._collect_facts(set(matched), set(selected), max_facts, hops)
        return {
            "query": query,
            "entities": [self.nodes[nid]["label"] for nid in matched],
            "clusters": [
                {"id": cid, "label": self.cluster_labels.get(cid, f"Cluster {cid}")}
                for cid in selected
            ],
            "facts": facts,
            "active_node_mask": cluster_mask(selected),
        }

    def _match_nodes(self, query, q_tokens):
        scores = Counter()
        labels = {nid: self.nodes[nid]["label"] for nid in self.nodes}
        for nid, label in labels.items():
            label_tokens = set(tokenize(label))
            overlap = q_tokens & label_tokens
            if overlap:
                scores[nid] += len(overlap) * 2
            if label and label in normalize_label(query):
                scores[nid] += 4
        vocabulary = {token: nid for nid, label in labels.items() for token in tokenize(label)}
        for token in q_tokens:
            for close in get_close_matches(token, vocabulary.keys(), n=2, cutoff=0.72):
                scores[vocabulary[close]] += 1.5
        return dict(scores.most_common(12))

    def _collect_facts(self, seeds, selected_clusters, max_facts, hops):
        adjacency = defaultdict(list)
        for edge in self.edges:
            adjacency[edge["source"]].append(edge)
            reverse = edge.copy()
            reverse["source"], reverse["target"] = edge["target"], edge["source"]
            reverse["_reverse"] = True
            adjacency[edge["target"]].append(reverse)

        queue = deque((nid, 0) for nid in seeds)
        seen = set(seeds)
        facts = []
        while queue and len(facts) < max_facts:
            nid, depth = queue.popleft()
            if depth >= hops:
                continue
            for edge in adjacency.get(nid, []):
                if selected_clusters and self.node_community.get(edge["target"]) not in selected_clusters and depth > 0:
                    continue
                src = self.nodes[edge["source"]]["label"]
                dst = self.nodes[edge["target"]]["label"]
                if not edge.get("_reverse"):
                    facts.append(f"{src} --{edge['relation']}--> {dst}")
                if edge["target"] not in seen:
                    seen.add(edge["target"])
                    queue.append((edge["target"], depth + 1))
                if len(facts) >= max_facts:
                    break
        return facts

    def to_dict(self):
        return {
            "version": "semantic_memory_v1",
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "communities": self.communities,
            "cluster_labels": self.cluster_labels,
        }

    @classmethod
    def from_dict(cls, data):
        graph = cls()
        for node in data.get("nodes", []):
            graph.nodes[node["id"]] = node
        graph.edges = data.get("edges", [])
        graph.communities = {str(k): v for k, v in data.get("communities", {}).items()}
        graph.node_community = {
            nid: str(cid)
            for cid, nodes in graph.communities.items()
            for nid in nodes
        }
        graph.cluster_labels = {str(k): v for k, v in data.get("cluster_labels", {}).items()}
        if graph.nodes and not graph.communities:
            graph.cluster()
        return graph

    def save(self, path=DEFAULT_MEMORY_PATH):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path=DEFAULT_MEMORY_PATH):
        if not Path(path).exists():
            graph = cls()
            graph.add_seed_facts()
            graph.cluster()
            return graph
        graph = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        graph.add_seed_facts()
        graph.cluster()
        return graph


def split_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def compact_phrase(text, max_words=6):
    words = tokenize(text)
    return " ".join(words[:max_words])


def cluster_mask(cluster_ids, num_nodes=64):
    if not cluster_ids:
        return [True] * num_nodes
    mask = [False] * num_nodes
    for cid in cluster_ids:
        try:
            cluster_num = int(cid)
        except ValueError:
            cluster_num = abs(hash(cid))
        start = (cluster_num * 7) % num_nodes
        width = max(4, num_nodes // 16)
        for offset in range(width):
            mask[(start + offset) % num_nodes] = True
    return mask


def build_memory_from_text(text):
    graph = SemanticMemoryGraph()
    graph.ingest_text(text)
    return graph
