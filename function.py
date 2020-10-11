import torch
def pairwise_distance(x1, x2, p=2, eps=1e-6):
    r"""
    Computes the batchwise pairwise distance between vectors v1,v2:

    .. math ::
        \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}

    Args:
        x1: first input tensor
        x2: second input tensor
        p: the norm degree. Default: 2
        eps (float, optional): Small value to avoid division by zero. Default: 1e-6

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.pairwise_distance(input1, input2, p=2)
        >>> output.backward()
    """
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff + eps, p).sum(dim=1, keepdim=True)
    return torch.pow(out, 1. / p)

def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    .. math::
        L(a, p, n) = \frac{1}{N} \left( \sum_{i=1}^N \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\} \right)

    where :math:`d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p`.

    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
        margin: the margin value. Default: 1
        p: the norm degree. Default: 2
        eps: small epsilon value to avoid numerical issues. Default: 1e-6
        swap: compute distance swap. Default: ``False``

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> input3 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.triplet_margin_loss(input1, input2, input3, p=2)
        >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Input must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    d_p = pairwise_distance(anchor, positive, p, eps)
    d_n = pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)
    loss = torch.mean(dist_hinge)
    return loss
	
def log_porb(a):
	x = (a>0.5).float()
	nx = 1 - x
	b = x*a + nx * (1-a)
	b = -b.log().sum(1)
	return b
