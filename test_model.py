from importnb import Notebook
with Notebook():
    from ML2_project import ConvolutionalLayer
import torch

# test for padding in ConvolutionalLayer
def test_padding():
    layer = ConvolutionalLayer(c_out=1, k=3, padding=1)

    x = torch.tensor([[[[1., 2.], [3., 4.]]]])
    
    x_padding = layer.padding_function(x)

    assert x_padding.shape == (1, 1, 4, 4) # assert output shape
    assert torch.equal(x_padding[:, :, 1:3, 1:3], x) # middle is x
    
    # borders are 0
    assert torch.all(x_padding[:, :, 0, :] == 0)
    assert torch.all(x_padding[:, :, -1, :] == 0)
    assert torch.all(x_padding[:, :, :, 0] == 0)
    assert torch.all(x_padding[:, :, :, -1] == 0)
