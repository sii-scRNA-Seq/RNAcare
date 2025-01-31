
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('IMID', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='sharedfileinstance',
            name='label',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
