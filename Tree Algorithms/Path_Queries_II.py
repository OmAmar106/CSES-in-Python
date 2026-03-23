import sys,os
from io import BytesIO, IOBase
BUFSIZE = (1<<13)
class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

n,q = (map(int, sys.stdin.readline().split()))
d = [[] for i in range(n)]
 
L = list(map(int, sys.stdin.readline().split()))
for _ in range(n-1):
    u,v = (map(int, sys.stdin.readline().split()))
    d[u-1].append(v-1)
    d[v-1].append(u-1)
 
parent = [-1]*n
depth = [0]*n
size = depth[:]
heavy = parent[:]
pos = depth[:]
stack = [(0, -1, 0)]
while stack:
    u, p, state = stack.pop()
    if state == 0:
        stack.append((u, p, 1))
        du = depth[u]
        for v in d[u]:
            if v != p:
                parent[v] = u
                depth[v] = du + 1
                stack.append((v, u, 0))
    else:
        size[u] = 1
        mx = 0
        pu = parent[u]
        for v in d[u]:
            if v != pu:
                sv = size[v]
                size[u] += sv
                if sv > mx:
                    mx = sv
                    heavy[u] = v

cur = 0
stack = [0]
while stack:
    u = stack.pop()
    h = u
    while u != -1:
        size[u] = h
        pos[u] = cur+n
        cur += 1
        hu = heavy[u]
        pu = parent[u]
        for v in d[u]:
            if v!=pu and v!=hu:
                stack.append(v)
        u = hu

seg = [0] * (2 * n)
for i in range(n):
    seg[pos[i]] = L[i]
for i in range(n - 1, 0, -1):
    seg[i] = seg[2*i] if seg[2*i] > seg[2*i+1] else seg[2*i+1]
 
def update(u, x):
    u = pos[u]
    seg[u] = x
    u >>= 1
    while u:
        seg[u] = seg[u<<1] if seg[u<<1]>seg[(u<<1)+1] else seg[(u<<1)+1]
        u >>= 1

def query(u, v):
    res = 0
    while size[u]!=size[v]:
        if depth[size[u]] < depth[size[v]]:
            u, v = v, u
        l = pos[size[u]]
        r = pos[u]+1
        while l < r:
            if l & 1: 
                if seg[l]>res:res = seg[l]
                l += 1
            if r & 1: 
                if seg[r-1]>res:
                    res = seg[r-1]
            l >>= 1; r >>= 1
        u = parent[size[u]]
 
    if depth[u] > depth[v]:
        u, v = v, u
 
    l = pos[u]
    r = pos[v]+1
    while l < r:
        if l & 1: 
            if seg[l]>res:res = seg[l]
            l += 1
        if r & 1: 
            if seg[r-1]>res:
                res = seg[r-1]
        l >>= 1; r >>= 1
 
    return res
 
ans = []
for _ in range(q):
    ty,s,x = (map(int, sys.stdin.readline().split()))
    if ty==1:
        update(s-1,x)
    else:
        ans.append(query(s-1,x-1))

print(' '.join(map(str,ans)))