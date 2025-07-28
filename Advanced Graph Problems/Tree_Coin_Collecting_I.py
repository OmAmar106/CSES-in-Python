import sys,math,cmath,random,os
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations,combinations
from io import BytesIO, IOBase
from decimal import Decimal,getcontext
 
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
    while b:
        a,b = b,a%b
    return a
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
II = lambda : int(sys.stdin.readline().strip())
LII = lambda : list(map(int, sys.stdin.readline().split()))
MI = lambda x : x(map(int, sys.stdin.readline().split()))
SI = lambda : sys.stdin.readline().strip()
SLI = lambda : list(map(lambda x:ord(x)-97,sys.stdin.readline().strip()))
LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
LII_C = lambda x : list(map(x, sys.stdin.readline().split()))
MATI = lambda x : [list(map(int, sys.stdin.readline().split())) for _ in range(x)]
##
 
#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare, SparseTable: SparseTable
#Bucket Sorted list: bsortl, Segment Tree(lp,selfop): SegmentTree, bootstrap: bootstrap, Trie: tries
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, Segment Tree(lp): SegmentOther
#Graph1(dnc,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, nummat: matrix, SuffixAutomaton: sautomaton
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file
 
def extras():
    getcontext().prec = 50
    sys.setrecursionlimit(10**6)
    sys.set_int_max_str_digits(10**5)
# extras()
 
def interactive():
    import builtins
    # print(globals())
    globals()['print'] = lambda *args, **kwargs: builtins.print(*args, flush=True, **kwargs)
# interactive()
 
def GI(n,m=None,sub=-1,dirs=False,weight=False):
    if m==None:
        m = n-1
    d = [[] for i in range(n)]
    if not weight:
        for i in range(m):
            u,v = LII_C(lambda x:int(x)+sub)
            d[u].append(v)
            if not dirs:
                d[v].append(u)
    else:
        for i in range(m):
            u,v,w = LII()
            d[u+sub].append((v+sub,w))
            if not dirs:
                d[v+sub].append((u+sub,w))
    return d
 
ordalp = lambda s : ord(s)-65 if s.isupper() else ord(s)-97
alp = lambda x : chr(97+x)
yes = lambda : print("Yes")
no = lambda : print("No")
yn = lambda flag : print("Yes" if flag else "No")
printf = lambda x : print(-1 if x==float('inf') else x)
lalp = 'abcdefghijklmnopqrstuvwxyz'
ualp = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
dirs = ((1,0),(0,1),(-1,0),(0,-1))
dirs8 = ((1,0),(0,1),(-1,0),(0,-1),(1,-1),(-1,1),(1,1),(-1,-1))
ldir = {'D':(1,0),'U':(-1,0),'R':(0,1),'L':(0,-1)}
 
INF = float("inf")
class Dinic:
    def __init__(self, n):
        self.lvl = [0] * n
        self.ptr = [0] * n
        self.q = [0] * n
        self.adj = [[] for _ in range(n)]
    def add_edge(self, a, b, c, rcap=0):
        self.adj[a].append([b, len(self.adj[b]), c, 0])
        self.adj[b].append([a, len(self.adj[a]) - 1, rcap, 0])
    def dfs(self, v, t, f):
        if v == t or not f:
            return f
        for i in range(self.ptr[v], len(self.adj[v])):
            e = self.adj[v][i]
            if self.lvl[e[0]] == self.lvl[v] + 1:
                p = self.dfs(e[0], t, min(f, e[2] - e[3]))
                if p:
                    self.adj[v][i][3] += p
                    self.adj[e[0]][e[1]][3] -= p
                    return p
            self.ptr[v] += 1
        return 0
    def calc(self, s, t):
        flow, self.q[0] = 0, s
        for l in range(31):  # l = 30 maybe faster for random data
            while True:
                self.lvl, self.ptr = [0] * len(self.q), [0] * len(self.q)
                qi, qe, self.lvl[s] = 0, 1, 1
                while qi < qe and not self.lvl[t]:
                    v = self.q[qi]
                    qi += 1
                    for e in self.adj[v]:
                        if not self.lvl[e[0]] and (e[2] - e[3]) >> (30 - l):
                            self.q[qe] = e[0]
                            qe += 1
                            self.lvl[e[0]] = self.lvl[v] + 1
                p = self.dfs(s, t, INF)
                while p:
                    flow += p
                    p = self.dfs(s, t, INF)
                if not self.lvl[t]:
                    break
        return flow
 
class binary_lift:
    def __init__(self, graph, f=max, root=0, flag=False,data=[]):
        n = len(graph)
        parent = [-1] * (n + 1)
        depth = self.depth = [-1] * n
        bfs = [root]
        depth[root] = 0;flag2 = True
        if not data:data = [0]*n;flag2 = False
        for node in bfs:
            if not flag2:
                for nei,w in graph[node]:
                    if depth[nei] == -1:
                        data[nei] = w
                        parent[nei] = node
                        depth[nei] = depth[node] + 1
                        bfs.append(nei)
            else:
                for nei in graph[node]:
                    if depth[nei] == -1:
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
    def kth_ancestor(self, a, k):
        parent = self.parent
        if self.depth[a] < k:
            return -1
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                a = parent[i][a]
        return a
    def __call__(self, a, b, c=0):
        depth = self.depth
        parent = self.parent
        data = self.data
        f = self.f
        c = self.lca(a, b)
        val = data[0][c]
        for x,d in (a, depth[a] - depth[c]), (b, depth[b] - depth[c]):
            for i in range(d.bit_length()):
                if (d >> i) & 1:
                    val = f(val, data[i][x])
                    x = parent[i][x]
        return val
    
def solve():
    n,q = LII()
    L = LII()
    d = GI(n)
 
    data = [-1]*n
 
    bfs = []
    for i in range(len(L)):
        if L[i]:
            data[i] = 0
            bfs.append((i,0))
        
    for elem,dist in bfs:
        for j in d[elem]:
            if data[j]==-1:
                data[j] = dist+1
                bfs.append((j,dist+1))
 
    gr = d
    n = len(gr)
    visited = [False]*n
    depth = [0]*n
    stack = [0]
    parent = [0]*n
    while stack:
        start = stack.pop()
        if not visited[start]:
            visited[start] = True
            for child in gr[start]:
                if not visited[child]:
                    depth[child] = depth[start]+1
                    parent[child] = start
                    stack.append(child)
 
    mini = []
    for i in range(n):
        mini.append(min(data[i],data[parent[i]]))
 
    mini = [mini]
    parent = [parent]
    
    for i in range(20):
        parent.append(parent[-1][:])
        mini.append(mini[-1][:])
        for j in range(n):
            parent[-1][j] = parent[-2][parent[-2][j]]
            mini[-1][j] = min(mini[-2][j],mini[-2][parent[-2][j]])
 
    def lca(a, b):
        if depth[a] < depth[b]:
            a, b = b, a
        d = depth[a] - depth[b]
        for i in range(20):
            if (d >> i) & 1:
                a = parent[i][a]
        if a == b:
            return a
        for i in reversed(range(20)):
            if parent[i][a] != parent[i][b]:
                a = parent[i][a]
                b = parent[i][b]
        return parent[0][a]
 
    def distf(a, b):
        c = lca(a, b)
        dist = depth[a] + depth[b] - 2 * depth[c]
        f = data[c]
        d = depth[a] - depth[c]
        x = a
        for i in range(20):
            if (d >> i) & 1:
                f = min(f, mini[i][x])
                x = parent[i][x]
        d = depth[b] - depth[c]
        x = b
        for i in range(20):
            if (d >> i) & 1:
                f = min(f, mini[i][x])
                x = parent[i][x]
        return dist + 2 * f
 
 
    for i in range(q):
        a,b = LII_1()
        # print(2*bl(a,b)+bl.distance(a,b))
        print(distf(a,b))
 
    #L1 = LII()
    #st = SI()
solve()