from __future__ import absolute_import

import numpy as np
import paddle
from paddle.fluid import layers
from paddle2onnx.constant import dtypes
from paddle2onnx.op_mapper import CustomPaddleOp, register_custom_paddle_op


class FillConstantBatchSizeLike(CustomPaddleOp):
    def __init__(self, node, **kw):
        super(FillConstantBatchSizeLike, self).__init__(node)

    def forward(self):
        input = self.input('Input', 0)
        input_shape = paddle.shape(input)
        updates = input_shape[self.node.attr('input_dim_idx')]
        shape = paddle.assign(np.array(self.node.attr('shape')).astype('int32'))
        dims = len(self.node.attr('shape'))
        new_shape = paddle.concat([
            shape[:self.node.attr('output_dim_idx')], updates,
            shape[self.node.attr('output_dim_idx') + 1:dims]
        ])
        dtype = dtypes.DTYPE_PADDLE_STR_MAP[self.node.attr('dtype')]
        out = paddle.full(new_shape, self.node.attr('value'), dtype)
        return {'Out': [out]}


register_custom_paddle_op('fill_constant_batch_size_like', FillConstantBatchSizeLike)