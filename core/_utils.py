def sum_to(NDArray, shape):
    diff = NDArray.ndim - len(shape)
    assert diff >= 0, 'cannot convert to bigger array'
    if diff != 0:
        for _ in range(diff):
            NDArray = NDArray.sum(axis=0)
    assert diff == 0, 're-write sum_to function'
    for i, dim in enumerate(shape):
        if dim == 1:
            NDArray = NDArray.sum(axis=i, keepdims=True)
    return NDArray

class PriorityQueue:
    '''Priorty Queue Implementation'''

    def __init__(self, data=None, comp_func=lambda x, y: x > y, key_func=None):
        if data is None:
            data = []
        self.size = len(data)
        self.data = data
        self.comp_func = comp_func
        self.key_func = key_func
        self._build_max_heap(self.comp_func)

    def __len__(self):
        return self.size

    def enqueue(self, data):
        self.data.append(data)
        self.size += 1
        cur_index = self.size

        while cur_index > 1 and self.comp_func(data, self.data[cur_index // 2]):
            self.data[cur_index] = self.data[cur_index // 2]
            cur_index = cur_index // 2

        self.data[cur_index] = data


    def dequeue(self):
        if self.size == 0:
            return None
        
        tmp = self.data[1]
        self.data[1] = self.data[self.size]
        self.size -= 1
        self.data.pop()

        if self.size > 0:
            self._max_heapify(1, self.comp_func)

        return tmp

    def max(self):
        return self.data[1]
    
    def empty(self):
        return True if self.size == 0 else False

    def increase_key(self, index, key):
        pass
        
    def _max_heapify(self, index, comp_func):
        largest_idx = index
        left_idx = 2 * index
        right_idx = 2 * index + 1

        if left_idx <= self.size and comp_func(self.data[left_idx], self.data[index]): # checks if left child exists
            largest_idx = left_idx
        
        if right_idx <= self.size and comp_func(self.data[right_idx], self.data[largest_idx]): # checks if right child exists
            largest_idx = right_idx

        if largest_idx != index:
            self.data[index], self.data[largest_idx] = self.data[largest_idx], self.data[index]
            self._max_heapify(largest_idx, comp_func)

    def _build_max_heap(self, comp_func):
        self.data.insert(0, None)
        for i in range(len(self.data) // 2, 0, -1):
            self._max_heapify(i, comp_func)

    def __repr__(self):
        return f'{self.data[1:]}'
