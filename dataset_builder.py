import datasets
from pathlib import Path


class GenerationsConfig(datasets.BuilderConfig):
    """BuilderConfig for SMHD."""

    def __init__(self, data_filepath, **kwargs):
        super(GenerationsConfig, self).__init__(**kwargs)
        self.data_filepath = Path(data_filepath)

class GenerationsDataset(datasets.GeneratorBasedBuilder):
    """SMHD - Self-Reported Mental Health Diagnoses."""

    BUILDER_CONFIG_CLASS = GenerationsConfig
    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features(
            {
                'text': datasets.Value('string'),
                'sentiment': datasets.Value('float32'),
            }
        )

        return datasets.DatasetInfo(
            description="",
            features=features,
            supervised_keys=None,
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': self.config.data_filepath,
                    'split': 'train',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'filepath': self.config.data_filepath,
                    'split': 'test'
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'filepath': self.config.data_filepath,
                    'split': 'validation'
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""

        # load data
        with open(filepath,'r') as f:
            lines = f.readlines()

        #spl_pt1=0.8
        #spl_pt2=0.9
        spl_pt1=1.
        spl_pt2=1.

        n_lines = len(lines)
        if split=='train':
            lines_to_iterate = range(0,int(n_lines*spl_pt1))
        elif split=='validation':
            lines_to_iterate = range(int(n_lines*spl_pt1), int(n_lines*spl_pt2))
        elif split=='test':
            lines_to_iterate = range(int(n_lines*spl_pt2), int(n_lines))

        for l in lines_to_iterate:
            line = lines[l].replace('\n','')
            text = line.split('r=')[0].strip()
            sentiment = float(line.split('r=')[-1])
            yield l, {'text': text, 'sentiment': sentiment}
