

from rlpyt.agents.pg.gaussian import (GaussianPgAgent,
    RecurrentGaussianPgAgent, AlternatingRecurrentGaussianPgAgent)
from rlpyt.cwto_models.cwto_atari_ff_model import CWTO_AtariFfModel
from rlpyt.cwto_models.cwto_atari_lstm_model import CWTO_AtariLstmModel


class CWTO_AtariMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    """

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract image shape and action size."""
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=2)


class CWTO_AtariFfAgent(CWTO_AtariMixin, GaussianPgAgent):

    def __init__(self, ModelCls=CWTO_AtariFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class CWTO_AtariLstmAgent(CWTO_AtariMixin, RecurrentGaussianPgAgent):

    def __init__(self, ModelCls=CWTO_AtariLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class CWTO_AlternatingAtariLstmAgent(CWTO_AtariMixin,
        AlternatingRecurrentGaussianPgAgent):

    def __init__(self, ModelCls=CWTO_AtariLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
