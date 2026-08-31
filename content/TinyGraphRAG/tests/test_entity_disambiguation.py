import copy
import unittest

from tinygraph.graph import TinyGraph


ENTITIES = [
    {
        "name": "Alice",
        "description": "A researcher",
        "chunks id": ["chunk-1"],
        "entity id": "entity-1",
    },
    {
        "name": "Alice",
        "description": "The same researcher",
        "chunks id": ["chunk-2"],
        "entity id": "entity-2",
    },
    {
        "name": "Bob",
        "description": "A colleague",
        "chunks id": ["chunk-1"],
        "entity id": "entity-3",
    },
]

TRIPLETS = [
    {
        "subject": "Alice",
        "subject_id": "entity-2",
        "predicate": "works with",
        "object": "Bob",
        "object_id": "entity-3",
    }
]


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def predict(self, prompt):
        self.prompts.append(prompt)
        return self.response


class EntityDisambiguationTest(unittest.TestCase):
    def make_graph(self, response):
        graph = TinyGraph.__new__(TinyGraph)
        graph.llm = FakeLLM(response)
        return graph

    def test_entity_ids_include_name_and_description(self):
        graph = self.make_graph(
            """
            <concept>
                <name>Alice</name>
                <description>A researcher</description>
            </concept>
            <concept>
                <name>Bob</name>
                <description>A researcher</description>
            </concept>
            """
        )

        entities = graph.get_entity("ignored", "chunk-1")

        self.assertNotEqual(entities[0]["entity id"], entities[1]["entity id"])

    def test_llm_mapping_updates_entities_and_triplets(self):
        graph = self.make_graph(
            '<transformation>{"entity-2": "entity-1"}</transformation>'
        )

        entities, triplets = graph._disambiguate_entities(
            copy.deepcopy(ENTITIES), copy.deepcopy(TRIPLETS), use_llm=True
        )

        self.assertEqual(
            ["entity-1", "entity-1", "entity-3"],
            [entity["entity id"] for entity in entities],
        )
        self.assertEqual("entity-1", triplets[0]["subject_id"])
        self.assertEqual("entity-3", triplets[0]["object_id"])
        self.assertEqual(1, len(graph.llm.prompts))
        self.assertIn("## Given Entities\n[", graph.llm.prompts[0])

    def test_none_response_keeps_distinct_ids(self):
        graph = self.make_graph("<transformation>None</transformation>")

        entities, triplets = graph._disambiguate_entities(
            copy.deepcopy(ENTITIES), copy.deepcopy(TRIPLETS), use_llm=True
        )

        self.assertEqual(
            ["entity-1", "entity-2", "entity-3"],
            [entity["entity id"] for entity in entities],
        )
        self.assertEqual("entity-2", triplets[0]["subject_id"])

    def test_unknown_ids_are_rejected(self):
        graph = self.make_graph(
            '<transformation>{"entity-2": "entity-unknown"}</transformation>'
        )

        with self.assertRaisesRegex(ValueError, "unknown entity IDs"):
            graph._disambiguate_entities(
                copy.deepcopy(ENTITIES), copy.deepcopy(TRIPLETS), use_llm=True
            )

    def test_ids_shared_by_different_names_are_rejected(self):
        graph = self.make_graph("")
        conflicting_entities = copy.deepcopy(ENTITIES)
        conflicting_entities[2]["entity id"] = "entity-1"

        with self.assertRaisesRegex(ValueError, "different entity names"):
            graph._disambiguate_entities(
                conflicting_entities, copy.deepcopy(TRIPLETS), use_llm=True
            )

        self.assertEqual([], graph.llm.prompts)

    def test_default_mode_merges_same_name_using_first_id(self):
        graph = self.make_graph("")

        entities, triplets = graph._disambiguate_entities(
            copy.deepcopy(ENTITIES), copy.deepcopy(TRIPLETS), use_llm=False
        )

        self.assertEqual(
            ["entity-1", "entity-1", "entity-3"],
            [entity["entity id"] for entity in entities],
        )
        self.assertEqual("entity-1", triplets[0]["subject_id"])
        self.assertEqual([], graph.llm.prompts)


if __name__ == "__main__":
    unittest.main()
