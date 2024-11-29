from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides 

    Returns:
        Position in storage ---> 一维数组中的位置
    """
    # TODO: Implement for Task 2.1.
    res = 0
    for i in range(len(index)):
        res += index[i]*strides[i]
    return res
    # raise NotImplementedError('Need to implement for Task 2.1')


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    for d in range(len(shape)-1,-1,-1):
        out_index[d] = ordinal % shape[d]
        ordinal = ordinal // shape[d]

def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # TODO: Implement for Task 2.2.
    # broadcast_index 函数根据广播规则将 big_shape 的 big_index 转换为较小的 shape 的 out_index。
    for i in range(len(shape)):
        offset = i + len(big_shape) - len(shape)
        out_index[i] = big_index[offset] if shape[i]!=1 else 0

def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # 如果遵守以下规则，则两个tensor是“可广播的”：
    # 每个tensor至少有一个维度；
    # 遍历tensor所有维度时，从末尾开始遍历（从右往左开始遍历）（从后往前开始遍历），两个tensor存在下列情况：
    # tensor维度相等。
    # tensor维度不等且其中一个维度为1。
    # tensor维度不等且其中一个维度不存在。
    # 1、判断两个tensor的维度是否相等,不相等需要在左侧进行 扩充
    l = max(len(shape1),len(shape2))
    if len(shape1) > len(shape2):
        # 扩充左边
        shape2 = [shape1[i] for i in range(l-len(shape2))] + list(shape2)
    else :
        shape1 = [shape2[i] for i in range(l-len(shape1))] + list(shape1)
    ans = []
    # 怎么进行扩充
    for i in range(l):
        # 不能扩充的情况
        if shape1[i]!=shape2[i] and shape1[i]!=1 and shape2[i]!=1:
            raise IndexingError('cannot broadcast')
        ans.append(max(shape1[i],shape2[i]))
    return tuple(ans)
    


    # TODO: Implement for Task 2.2.
    # raise NotImplementedError('Need to implement for Task 2.2')


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """
        storage: 存储的连续数据
        shape: tensor被认为的形状
        strides:对于一个多维数组或张量，strides提供了一种方式来理解如何从一个元素移动到另一个元素。在多维数组中，strides的每个元素表示为了到达下一个元素，需要在内存中前进多少个元素的字节。例如，对于一个二维数组，strides的第一个元素表示在行内移动到下一个元素需要的字节数，第二个元素表示移动到下一行的下一个元素需要的字节数。
        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            # 从默认的形状中的得到每个维度的步长
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)  # dims ： 一共几个维度
        self.size = int(prod(shape)) # size :一共多少个元素
        self.shape = shape 
        assert len(self._storage) == self.size # 原始的一维数据 要等于 size的大小

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1.
        new_shape = []
        new_stride = []
        for i in  order:
            new_shape.append(self.shape[i])
        for i in  order:
            new_stride.append(self.strides[i])
        return TensorData(self._storage,tuple(new_shape),tuple(new_stride))

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
