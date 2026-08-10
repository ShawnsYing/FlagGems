from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts

# mkldnn_rnn_layer is LSTM (mode=2), single-layer, unidirectional.
_MODE = 2
_NUM_LAYERS = 1
_BIDIRECTIONAL = False
_BATCH_FIRST = False


class MkldnnRnnLayerBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes = [
            (16, 4, 32),
            (32, 8, 64),
            (64, 16, 128),
        ]
        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


def mkldnn_rnn_layer_input_fn(shape, dtype, device):
    seq_len, batch_size, input_size = shape
    hidden_size = input_size
    inp = torch.randn(seq_len, batch_size, input_size, dtype=dtype, device=device)
    lstm = torch.nn.LSTM(input_size, hidden_size, 1).to(dtype=dtype, device=device)
    w_ih = lstm.weight_ih_l0
    w_hh = lstm.weight_hh_l0
    b_ih = lstm.bias_ih_l0
    b_hh = lstm.bias_hh_l0
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    # All arguments are positional in the aten schema:
    # (input, w0, w1, w2, w3, hx, cx, reverse, batch_sizes, mode, hidden_size,
    #  num_layers, has_biases, bidirectional, batch_first, train)
    yield (
        inp,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hx,
        cx,
        False,
        [],
        _MODE,
        hidden_size,
        _NUM_LAYERS,
        True,
        _BIDIRECTIONAL,
        _BATCH_FIRST,
        False,
    )


def _torch_mkldnn_rnn_layer_cpu_baseline(inp, w_ih, w_hh, b_ih, b_hh, hx, cx, *rest):
    """torch.mkldnn_rnn_layer (oneDNN) is CPU-only, so route the tensor inputs
    through CPU and move the results back to the original device."""
    dev = inp.device
    cpu = [t.detach().to("cpu") for t in (inp, w_ih, w_hh, b_ih, b_hh, hx, cx)]
    out, hy, cy, ws = torch.mkldnn_rnn_layer(*cpu, *rest)
    return out.to(dev), hy.to(dev), cy.to(dev), ws


@pytest.mark.mkldnn_rnn_layer
def test_mkldnn_rnn_layer():
    bench = MkldnnRnnLayerBenchmark(
        input_fn=mkldnn_rnn_layer_input_fn,
        op_name="mkldnn_rnn_layer",
        torch_op=_torch_mkldnn_rnn_layer_cpu_baseline,
        gems_op=flag_gems.ops.mkldnn_rnn_layer,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
