class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.data = []
        self.current = None

    def full(self):
        return len(self.data) == self.size

    def append(self, item):
        if not self.full():
            self.data.append(item)
        else:
            if self.current is None:
                self.current = 0
            self.data[self.current] = item
            self.current = (self.current + 1) % self.size
    
    def get(self):
        if self.current is None:
            current = 0
        else:
            current = self.current
        return self.data[current:] + self.data[:current]

    def __getitem__(self, slice):
        return self.get()[slice]

    def __len__(self):
        return len(self.data)
