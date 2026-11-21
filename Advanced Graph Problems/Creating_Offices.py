import sys,math,cmath,random,os
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

II = lambda : int(sys.stdin.readline().strip())
LII = lambda : list(map(int, sys.stdin.readline().split()))
LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))

import gc
gc.disable()

class binary_lift:
    def __init__(self, graph, f=max, root=0, flag=False):
        n = len(graph)
        parent = [-1] * (n + 1)
        depth = self.depth = [-1] * n
        bfs = [root]
        depth[root] = 0
        data = [0]*n
        for node in bfs:
            # for nei,w in graph[node]:
            for nei in graph[node]:
                if depth[nei] == -1:
                    # data[nei] = w
                    parent[nei] = node
                    depth[nei] = depth[node] + 1
                    bfs.append(nei)
        parent = self.parent = [parent]
        self.f = f
        if flag:
            data = self.data = [data]
            for _ in range(max(depth).bit_length()):
                old_data = data[-1]
                old_parent = parent[-1]
                data.append([f(val, old_data[p]) for val,p in zip(old_data, old_parent)])
                parent.append([old_parent[p] for p in old_parent])
        else:
            for _ in range(max(depth).bit_length()):
                old_parent = parent[-1]
                parent.append([old_parent[p] for p in old_parent])
    def lca(self, a, b):
        depth = self.depth
        parent = self.parent
        if depth[a] < depth[b]:
            a,b = b,a
        d = depth[a] - depth[b]
        for i in range(d.bit_length()):
            if (d >> i) & 1:
                a = parent[i][a]
        for i in range(depth[a].bit_length())[::-1]:
            if parent[i][a] != parent[i][b]:
                a = parent[i][a]
                b = parent[i][b]
        if a != b:
            return parent[0][a]
        else:
            return a
    def distance(self, a, b):
        return self.depth[a] + self.depth[b] - 2 * self.depth[self.lca(a,b)]

def shallowest_decomposition_tree(graph, root=0):
    # returns a root and a directed tree, iterate on it and you
    # will find the new centroid
    # slightly faster than centroid_decomposition_tree
    ctz = lambda x:(x & -x).bit_length()-1
    n = len(graph)
    forbidden = [0] * n
    decomposition_tree = [-1]*n
    stacks = [[] for _ in range(n.bit_length())]
    def extract_chain(labels, u):
        while labels:
            label = labels.bit_length() - 1
            labels ^= (1<<label)
            v = stacks[label].pop()
            decomposition_tree[v] = u
            u = v
    dfs = [root]
    while dfs:
        u = dfs.pop()
        if u >= 0:
            forbidden[u] = -1
            dfs.append(~u)
            for v in graph[u]:
                if not forbidden[v]:
                    dfs.append(v)
        else:
            u = ~u
            forbidden_once = forbidden_twice = 0
            for v in graph[u]:
                forbidden_twice  |= forbidden_once & (forbidden[v] + 1)
                forbidden_once  |= forbidden[v] + 1
            forbidden[u] = forbidden_once | ((1<<forbidden_twice.bit_length()) - 1)
            label_u = ctz(forbidden[u] + 1)
            stacks[label_u].append(u)
            for v in graph[u]:
                extract_chain((forbidden[v] + 1) & ((1<<label_u) - 1), u)
    max_label = (forbidden[root] + 1).bit_length() - 1
    decomposition_root = stacks[max_label].pop()
    extract_chain((forbidden[root] + 1) & ((1<<max_label) - 1), decomposition_root)
    return decomposition_tree, decomposition_root

"""

need some ds in which i can insert nodes
and query the closest node in a set to a particular node
for this, need centroid decomposition


Iterate over the parents, since 
every node has at max log n parent, for every parent
store the closest node to it in its subtree
and while checking closest node, iterate on its parent

"""

def solve():
    n,k = LII()

    d = [[] for i in range(n)]
    for u in range(1,n):
        u,v = LII_1()
        d[u].append(v)
        d[v].append(u)

    bl = binary_lift(d)

    def dist1(a,b):
        return bl.distance(a,b)
    
    d,root = shallowest_decomposition_tree(d)

    best = [float('inf')]*n

    def query(val):
        ans = best[val]+dist1(val,val)
        val1 = val
        while val!=root:
            val = d[val]
            ans = min(ans,best[val]+dist1(val,val1))
        return ans

    def update(val):
        best[val] = 0
        val1 = val
        while val!=root:
            val = d[val]
            best[val] = min(best[val],dist1(val,val1))
        return

    ans = []
    for i in sorted(range(n),key=lambda x:-bl.depth[x]):
        dist = query(i)
        if dist>=k:
            ans.append(i+1)
            update(i)

    print(len(ans))
    print(' '.join(map(str,ans)))

    #L1 = LII()
    #st = SI()
solve()