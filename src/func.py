import numpy as np

from typing import List, Union

from symai import core_ext, Interface, Import
from symai import Symbol, Expression, Metadata
from symai.components import Embed, Metric


class RewardModel(Expression):
    def __init__(self, in_memory=True, metric='jaccard', aggregation: Union['sum', 'mean', 'median'] = 'sum'):
        super().__init__()
        self.in_memory   = in_memory
        self.sim_metric  = metric
        self.aggregation = aggregation
        self.metric      = Metric(normalize=True)
        self.model       = Interface('llava')
        self.clip        = Interface('clip')
        self.embed       = Import('ExtensityAI/embeddings')
        #self.embed       = Embed()

    def _dynamic_cache(self, references: List[str]):
        @core_ext.cache(in_memory=self.in_memory)
        def _embed(_):
            def _vision_(image):
                desc = self.model(image=image, query='Describe the image')
                emb  = self.embed(desc)
                emb  = Symbol(emb)
                return emb
            emb = map(_vision_, references)
            return list(emb)
        return _embed(self)

    def forward(self, images: List[str], references: List[str],  *args, **kwargs):
        # embed and cache references
        reference_embs = self._dynamic_cache(references)

        # embed images
        img_embeddings = {}
        for image in images:
            desc  = self.model(image=image, query='Describe the image')
            emb   = self.embed(desc)
            sym   = Symbol(emb)
            img_embeddings[str(image)] = sym
        joined_similarities = {k: None for k in img_embeddings.keys()}

        # calculate similarity
        for image, img_emb in img_embeddings.items():
            similarity    = [img_emb.similarity(emb, metric=self.sim_metric) for emb in reference_embs]
            sym           = Symbol(similarity)
            sym._metadata = img_emb._metadata
            joined_similarities[image] = sym

        # sum scores and normalize
        scores = []
        metas  = []
        for image, sim in joined_similarities.items():
            agg_func = np.sum
            if self.aggregation == 'mean':
                agg_func = np.mean
            elif self.aggregation == 'median':
                agg_func = np.median
            else:
                raise ValueError(f'Aggregation function "{self.aggregation}" is not supported.')
            score = agg_func(sim.value)
            scores.append(score)
            metas.append(sim._metadata)

        sym = self.metric(scores)
        sym._metadata = Metadata()
        sym._metadata.scores        = scores
        sym._metadata.similarities  = joined_similarities
        sym._metadata.results       = metas

        return sym


if __name__ == '__main__':
    expr = RewardModel()
    res = expr('assets/img/cat.jpeg')
    print(res)
