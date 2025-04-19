import sys,math,cmath,random,os
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations
from io import BytesIO, IOBase

BUFSIZE = 8192
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

# functions #
# MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(1,2**62)
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
##

#String hashing: sh/shclass, fenwick sortedlist: fsortl, Number: numtheory, SparseTable: SparseTable
#Bucket Sorted list: bsortl, Segment Tree(lazy propogation): SegmentTree/Other, bootstrap: bootstrap
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
#Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
#Persistent Segment Tree: perseg, FreqGraphs: bgraph, Binary Trie: b_trie, XOR_dict: xdict
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

class SegmentTree:
    @staticmethod
    def func(a, b):
        return max(a, b)
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        self.build(data)
    def build(self, data):
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.func(self.tree[i * 2], self.tree[i * 2 + 1])
    def update(self, pos, value):
        pos += self.n
        self.tree[pos] = value
        while pos > 1:
            pos //= 2
            self.tree[pos] = self.func(self.tree[2 * pos], self.tree[2 * pos + 1])
    def query(self, left, right):
        # Query the maximum value in the range [left, right)
        left += self.n
        right += self.n
        # Change the initializer depending upon the self.func
        max_val = float('-inf')
        ##
        while left < right:
            if left % 2:
                max_val = self.func(max_val, self.tree[left])
                left += 1
            if right % 2:
                right -= 1
                max_val = self.func(max_val, self.tree[right])
            left //= 2
            right //= 2
        return max_val
    def __repr__(self):
        print('Seg[',end='')
        for i in range(self.n):
            if i!=self.n-1:
                print(self.query(i,i+1),end=', ')
            else:
                print(self.query(i,i+1),end=']')
        print()

class HLD:
    def __init__(self, adj, values, root=0):
        self.n = len(adj)
        self.adj = adj
        self.values = values
        self.root = root
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        self.size = [0] * self.n
        self.heavy = [-1] * self.n
        self.head = [0] * self.n
        self.pos = [0] * self.n
        self.flat = [0] * self.n
        self.time = 0
        self._dfs(self.root)
        self._decompose(self.root, self.root)
        self.seg = SegmentTree([self.values[self.flat[i]] for i in range(self.n)])
    def _dfs(self,start=0):
        graph = self.adj
        n = self.n
        visited = [False] * n
        stack = [start]
        while stack:
            start = stack[-1]
            if not visited[start]:
                visited[start] = True
                for child in graph[start]:
                    if not visited[child]:
                        self.parent[child] = start
                        self.depth[child] = self.depth[start]+1
                        stack.append(child)
            else:
                stack.pop()
                self.size[start] = 1
                k = 0
                for child in graph[start]:
                    if self.parent[start]!=child:
                        self.size[start] += self.size[child]
                        if self.size[child]>k:
                            k = self.size[child]
                            self.heavy[start] = child
        return visited
    def _decompose(self, root, h):
        stack = [(root, h)]
        while stack:
            u, h = stack.pop()
            self.head[u] = h
            self.flat[self.time] = u
            self.pos[u] = self.time
            self.time += 1
            for v in reversed(self.adj[u]):
                if v != self.parent[u] and v != self.heavy[u]:
                    stack.append((v, v))
            if self.heavy[u] != -1:
                stack.append((self.heavy[u], h))
    def query(self, u, v):
        res = float('-inf') # update this depending upon the func
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            res = SegmentTree.func(res, self.seg.query(self.pos[self.head[u]], self.pos[u] + 1))
            u = self.parent[self.head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        res = SegmentTree.func(res, self.seg.query(self.pos[u], self.pos[v] + 1))
        return res
    def update(self, u, value):
        self.seg.update(self.pos[u], value)


def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    d = [[] for i in range(n)]
    for i in range(n-1):
        u,v = list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
        d[u].append(v)
        d[v].append(u)
    hld = HLD(d,L)
    for i in range(q):
        t,s,x = list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
        if t==0:
            hld.update(s,x+1)
        else:
            print(hld.query(s,x),end=' ')
    #st = sys.stdin.readline().strip()
solve()
