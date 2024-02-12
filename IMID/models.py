from django.db import models


class Gene(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50)


class GOTerm(models.Model):
    name = models.CharField(max_length=50, unique=True)
    term = models.CharField(max_length=255, blank=True, null=True)
    gene = models.ManyToManyField(Gene, related_name="go_term")
