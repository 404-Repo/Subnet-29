import numpy as np

from typing import List, Union

from symai import core_ext, Interface
from symai import Symbol, Expression, Metadata
from symai.components import Metric


class RewardModel(Expression):
    def __init__(self, in_memory=True, metric='cosine', aggregation: Union['sum', 'mean', 'median', 'none'] = 'sum'):
        super().__init__()
        self.in_memory   = in_memory
        self.sim_metric  = metric
        self.aggregation = aggregation
        self.metric      = Metric(normalize=True)
        self.model       = Interface('llava')
        self.clip        = Interface('clip')
        self.embed       = Interface('ExtensityAI/embeddings')

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

    def forward(self, images: List[str], refs_images: List[str], refs_texts: List[str], *args, **kwargs):
        # embed and cache references
        reference_embs = self._dynamic_cache(refs_images)

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
            elif self.aggregation == 'none':
                agg_func = lambda x: x
            else:
                raise ValueError(f'Aggregation function "{self.aggregation}" is not supported.')
            score = agg_func(sim.value)
            scores.append(score)
            metas.append(sim._metadata)

        sym1  = self.metric(scores)
        sym1._metadata = Metadata()
        sym1._metadata.scores        = scores
        sym1._metadata.similarities  = joined_similarities
        sym1._metadata.results       = metas

        sym2     = self.clip(images, refs_texts)
        indices  = sym2.value.argmax(axis=1)
        max_vals = sym2.value.max(axis=1)
        max_vals[indices == len(refs_texts) - 1] = 0
        s_comb   = (sym1.value.squeeze() + max_vals)
        rank     = np.exp(s_comb) / np.sum(np.exp(s_comb))

        res      = Symbol(rank)
        res._metadata        = Metadata()
        res._metadata.sym1   = sym1
        res._metadata.sym2   = sym2
        res._metadata.scores = s_comb
        res._metadata.rank   = rank
        return res


if __name__ == '__main__':
    expr = RewardModel()
    res = expr('assets/img/cat.jpeg')
    print(res)
