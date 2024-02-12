from functools import cache

from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag

from genes_ncbi_proteincoding import GENEID2NT


@cache
def build_GOEnrichmentStudyNS():
    obo_fname = download_go_basic_obo()
    godag = GODag(obo_fname)
    gene2go_fname = download_ncbi_associations()
    gene2go_reader = Gene2GoReader(gene2go_fname, taxids=[9606])
    ns2assoc = gene2go_reader.get_ns2assc()  # bp,cc,mf
    return GOEnrichmentStudyNS(GENEID2NT.keys(), ns2assoc, godag, propagate_counts=False, alpha=.05, methods=['fdr_bh'])
