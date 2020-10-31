
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel


class CWTO_AtariFfModel(torch.nn.Module):
    """
    Feedforward model for Atari agents: a convolutional network feeding an
    MLP with outputs for action probabilities and state-value estimate.
    """

    def __init__(
            self,
            image_shape,
            output_size=2,
            fc_sizes=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [16, 32],
            kernel_sizes=kernel_sizes or [8, 4],
            strides=strides or [4, 2],
            paddings=paddings or [0, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )
        self._output_size = output_size
        self.pi = torch.nn.Linear(self.conv.output_size, output_size*2)
        self.value = torch.nn.Linear(self.conv.output_size, 1)

    def forward(self, image, prev_action, prev_reward):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        *image_shape], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = self.pi(fc_out)
        mu = pi[:,:self._output_size]
        log_std = pi[:,self._output_size:-1]
        v = self.value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, v = restore_leading_dims((mu,log_std, v), lead_dim, T, B)

        return mu, log_std, v
