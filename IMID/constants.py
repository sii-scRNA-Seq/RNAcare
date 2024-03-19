from collections import namedtuple

Ontology = namedtuple("Ontology", ["name", "color"])

ONTOLOGY = {
    "BP": Ontology(name="biological_process", color="blues_r"),
    "CC": Ontology(name="cellular_component", color="purp_r"),
    "MF": Ontology(name="molecular_function", color="oranges_r"),
}

GeneID_URL = "https://biotools.fr/human/ensembl_symbol_converter/"
BUILT_IN_LABELS = {
    "obs",
    "FileName",
    "LABEL",
    "cluster",
    "batch1",
    "batch2",
}
NUMBER_CPU_LIMITS = 2
