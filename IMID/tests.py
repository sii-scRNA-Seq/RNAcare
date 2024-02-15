import time

from django.test import TestCase
import pandas as pd

from .models import Gene, GOTerm
from .utils import build_GOEnrichmentStudyNS
from genes_ncbi_proteincoding import GENEID2NT


class GoEnrichmentTimingTestCase(TestCase):
    fixtures = ['gene.json',]

    def setUp(self):
        # select list of test_genes for both methods (list of gene name strings)
        self.num_test_genes = 10000
        self.test_genes = [g.name for g in Gene.objects.order_by('?')[:self.num_test_genes]]

        # pre-build GOEnrichmentStudyNS object - same method for both new and old
        self.goeaobj = build_GOEnrichmentStudyNS()

        # pre-processing for old method
        self.mapper = {}
        for key in GENEID2NT:
            self.mapper[GENEID2NT[key].Symbol] = GENEID2NT[key].GeneID
        self.inv_map = {v: k for k, v in self.mapper.items()}

        self.GO_items = []
        temp = self.goeaobj.ns2objgoea["BP"].assoc  # BIOLOGICAL_PROCESS
        for item in temp:
            self.GO_items += temp[item]
        temp = self.goeaobj.ns2objgoea["CC"].assoc  # CELLULAR COMPONENT
        for item in temp:
            self.GO_items += temp[item]
        temp = self.goeaobj.ns2objgoea["MF"].assoc  # MOLECULAR_FUNCTION
        for item in temp:
            self.GO_items += temp[item]

    def test_run_new_method_timing(self):
        start = time.time()
        goea_results_all = self.goeaobj.run_study([g.id for g in Gene.objects.filter(name__in=self.test_genes)])
        goea_result_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05 and r.ratio_in_study[0] > 1]
        db_backed_run_study_seconds = time.time() - start

        start = time.time()
        go_df = pd.DataFrame(
            data=map(
                lambda x: [
                    x.goterm.name,
                    x.goterm.namespace,
                    x.p_fdr_bh,
                    x.ratio_in_study[0],
                    GOTerm.objects.get(name=x.GO).gene.count()
                ],
                goea_result_sig,
            ),
            columns=[
                "term",
                "class",
                "p_corr",
                "n_genes",
                "n_go",
            ],
            index=map(lambda x: x.GO, goea_result_sig)
        )
        go_df["per"] = go_df.n_genes / go_df.n_go
        db_backed_go_df_seconds = time.time() - start
        print("ELAPSED TIME (s) - {} test genes:".format(self.num_test_genes))
        print("new db-backed method: {}".format(db_backed_run_study_seconds + db_backed_go_df_seconds))

    def test_run_old_method_timing(self):
        start = time.time()
        mapped_genes = []
        for gene in self.test_genes:
            if gene in self.mapper:
                mapped_genes.append(self.mapper[gene])
        goea_results_all = self.goeaobj.run_study(mapped_genes)
        goea_result_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
        mem_backed_run_study_seconds = time.time() - start

        start = time.time()
        GO = pd.DataFrame(
            list(
                map(
                    lambda x: [
                        x.GO,
                        x.goterm.name,
                        x.goterm.namespace,
                        x.p_uncorrected,
                        x.p_fdr_bh,
                        x.ratio_in_study[0],
                        x.ratio_in_study[1],
                        self.GO_items.count(x.GO),
                        list(map(lambda y: self.inv_map[y], x.study_items)),
                    ],
                    goea_result_sig,
                )
            ),
            columns=[
                "GO",
                "term",
                "class",
                "p",
                "p_corr",
                "n_genes",
                "n_study",
                "n_go",
                "study_genes",
            ],
        )
        GO = GO[GO.n_genes > 1]
        mem_backed_go_df_seconds = time.time() - start
        print("ELAPSED TIME (s) - {} test genes:".format(self.num_test_genes))
        print("old memory-backed method: {}".format(mem_backed_run_study_seconds + mem_backed_go_df_seconds))
