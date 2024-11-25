from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # 计算函数f 在逼近arg时候的导数
    # 拆分输入 
    inps = [inp for inp in vals]
    # 计算在arg 和 arg+epsilon的函数时
    inps[arg] += epsilon
    neg = [inp for inp in vals]
    neg[arg] -= epsilon
    # 导数的近似
    return (f(*inps)-f(*neg))/(2*epsilon)
    
        
    
    


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # 基于深度优先搜索的算法 来进行拓扑排序
    # 根据输出的结点来进行拓扑排序
    PermanentMarked = []
    TemporaryMarked = []
    
    result = []
    
    def dfs(variable):
        if variable.is_constant():
            # 不处理常量
            return
        if variable.unique_id in PermanentMarked:
            # 这个结点已经被处理
            return 
        elif variable.unique_id in TemporaryMarked:
            raise(RuntimeError("Not a DAG"))
        # 开始正常处理结点的排序 ，从后往前排
        # 这个结点没处理完的时候，是需要标记的
        TemporaryMarked.append(variable.unique_id)
        if variable.is_leaf():
            pass #为啥不直接返回,因为需要移除标记位
        else :
            # 遍历结点
            for inp in variable.history.inputs:
                dfs(inp) # 遍历输入
            
        TemporaryMarked.remove(variable.unique_id)
        PermanentMarked.append(variable.unique_id)

        # 将拓扑的结果插入到res的list里面
        result.insert(0,variable)
    dfs(variable=variable)
    return result
    
    # raise NotImplementedError('Need to implement for Task 1.4')

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # 第一步获取拓扑序列
    res = topological_sort(variable)
    derivs = {variable.unique_id:deriv} # 记录当前变量的导数
       # 根据拓扑排序进行 反向传播-> 流程：反向传播-> 链式法则 --> 汇总导数，只有在叶子结点才会有存储梯度
    for node in res:
        # print(node)
        d_output = derivs[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(d_output)
        else :
            res_chain_relu = node.chain_rule(d_output)
            for inp, d in res_chain_relu:
                if inp.unique_id not in derivs:
                    derivs[inp.unique_id] = 0.0
                derivs[inp.unique_id]+=d



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
