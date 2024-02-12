from django.db import migrations

from genes_ncbi_proteincoding import GENEID2NT
from IMID.constants import ONTOLOGY
from IMID.utils import build_GOEnrichmentStudyNS


def create_genes(apps, schema_editor):
    Gene = apps.get_model("IMID", "Gene")
    for key in GENEID2NT:
        Gene.objects.get_or_create(id=GENEID2NT[key].GeneID, name=GENEID2NT[key].Symbol)


def delete_genes(apps, schema_editor):
    Gene = apps.get_model("IMID", "Gene")
    Gene.objects.all().delete()


def create_go_terms(apps, schema_editor):
    GOTerm = apps.get_model("IMID", "GOTerm")
    Gene = apps.get_model("IMID", "Gene")
    goea_obj = build_GOEnrichmentStudyNS()
    for o in ONTOLOGY:  # biological_process, cellular_component, molecular_function
        for gene_id, go_terms in goea_obj.ns2objgoea[o].assoc.items():
            try:
                gene = Gene.objects.get(id=gene_id)
            except Gene.DoesNotExist:
                continue
            for go in go_terms:
                go_term, _ = GOTerm.objects.get_or_create(name=go)
                go_term.gene.add(gene)


def delete_go_terms(apps, schema_editor):
    GOTerm = apps.get_model("IMID", "GOTerm")
    GOTerm.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('IMID', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(code=create_genes, reverse_code=delete_genes),
        migrations.RunPython(code=create_go_terms, reverse_code=delete_go_terms),
    ]
