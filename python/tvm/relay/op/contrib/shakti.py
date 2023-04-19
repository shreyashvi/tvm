import tvm.ir
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table


# def _register_external_op_helper(op_name, supported=True):
   
#     @tvm.ir.register_op_attr(op_name, "target.shakti")
#     def _func_wrapper(attrs, args):
#         return supported

#     return _func_wrapper



# _register_external_op_helper("nn.conv2d")
# _register_external_op_helper("nn.relu")
# _register_external_op_helper("add")
# _register_external_op_helper("subtract")
# _register_external_op_helper("multiply")

@tvm.ir.register_op_attr("nn.conv2d", "target.shakti")
def _my_conv2d_wrapper(expr):
    return True

@tvm.ir.register_op_attr("nn.relu", "target.shakti")
def _my_relu_wrapper(expr):
    return True

@tvm.ir.register_op_attr("nn.multiply", "target.shakti")
def _my_multiply_wrapper(expr):
    return True
@tvm.ir.register_op_attr("nn.add", "target.shakti")
def _my_add_wrapper(expr):
    return True

@tvm.ir.register_op_attr("nn.subtract", "target.shakti")
def _my_subtract_wrapper(expr):
    return True
