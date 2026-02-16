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
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

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

def reduce_time():
    # reduces run time, but increases the change of mle
    import gc
    gc.disable()
# reduce_time()

# ordalp = lambda s : ord(s)-65 if s.isupper() else ord(s)-97
# alp = lambda x : chr(97+x)
# yes = lambda : print("Yes")
# no = lambda : print("No")
# yn = lambda flag : print("Yes" if flag else "No")
# printf = lambda x : print(-1 if x==float('inf') else x)
# lalp = 'abcdefghijklmnopqrstuvwxyz'
# ualp = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# dirs = ((1,0),(0,1),(-1,0),(0,-1))
# dirs8 = ((1,0),(0,1),(-1,0),(0,-1),(1,-1),(-1,1),(1,1),(-1,-1))
# ldir = {'D':(1,0),'U':(-1,0),'R':(0,1),'L':(0,-1)}

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

class HLPP:
    def __init__(self,n):
        self.n=n
        self.g=[[] for _ in range(n)]
    def add_edge(self,u,v,c):
        self.g[u].append([v,c,len(self.g[v])])
        self.g[v].append([u,0,len(self.g[u])-1])
    def maxflow(self,s,t):
        n,g=self.n,self.g
        h=[0]*n; ex=[0]*n
        seen=[0]*n
        h[s]=n
        for v,c,rev in g[s]:
            if c:
                g[v][rev][1]+=c
                ex[v]+=c
        q=deque([i for i in range(n) if i!=s and i!=t and ex[i]>0])
        def push(u,e):
            v,c,rev=e
            d=min(ex[u],c)
            if h[u]==h[v]+1 and d>0:
                e[1]-=d; g[v][rev][1]+=d
                ex[u]-=d; ex[v]+=d
                if ex[v]==d and v!=s and v!=t:
                    q.append(v)
                return True
            return False
        def relabel(u):
            h[u]=min(h[v]+1 for v,c,_ in g[u] if c>0)
        while q:
            u=q[0]
            while ex[u]:
                if seen[u]<len(g[u]):
                    if not push(u,g[u][seen[u]]):
                        seen[u]+=1
                else:
                    relabel(u)
                    seen[u]=0
            q.popleft()
        return ex[t]


class AuxiliaryTree:
    def __init__(self, edge, root = 0):
        self.n = len(edge)
        self.order = [-1] * self.n
        self.path = [-1] * (self.n-1)
        self.depth = [0] * self.n
        if self.n == 1: return
        parent = [-1] * self.n
        que = [root]
        t = -1
        while que:
            u = que.pop()
            self.path[t] = parent[u]
            t += 1
            self.order[u] = t
            for v in edge[u]:
                if self.order[v] == -1:
                    que.append(v)
                    parent[v] = u
                    self.depth[v] = self.depth[u] + 1
        self.n -= 1
        self.h = self.n.bit_length()
        self.data = [0] * (self.n * self.h)
        self.data[:self.n] = [self.order[u] for u in self.path]
        for i in range(1, self.h):
            for j in range(self.n - (1<<i) + 1):
                self.data[i*self.n + j] = min(self.data[(i-1)*self.n + j], self.data[(i-1)*self.n + j+(1<<(i-1))])

    def lca(self, u, v):
        if u == v: return u
        l = self.order[u]
        r = self.order[v]
        if l > r:
            l,r = r,l
        level = (r - l).bit_length() - 1
        return self.path[min(self.data[level*self.n + l], self.data[level*self.n + r-(1<<level)])]

    def dis(self, u, v):
        if u == v: return 0
        l = self.order[u]
        r = self.order[v]
        if l > r:
            l,r = r,l
        level = (r - l).bit_length() - 1
        p = self.path[min(self.data[level*self.n + l], self.data[level*self.n + r-(1<<level)])]
        return self.depth[u] + self.depth[v] - 2 * self.depth[p]

    def make(self, vs):
        k = len(vs)
        vs.sort(key = self.order.__getitem__)

        par = dict()
        edge = dict()
        edge[vs[0]] = []

        st = [vs[0]]

        for i in range(k - 1):
            l = self.order[vs[i]]
            r = self.order[vs[i+1]]
            level = (r - l).bit_length() - 1
            w = self.path[min(self.data[level*self.n + l], self.data[level*self.n + r-(1<<level)])]
            if w != vs[i]:
                p = st.pop()
                while st and self.depth[w] < self.depth[st[-1]]:
                    par[p] = st[-1]
                    edge[st[-1]].append(p)
                    p = st.pop()

                if not st or st[-1] != w:
                    st.append(w)
                    edge[w] = [p]
                else:
                    edge[w].append(p)
                par[p] = w

            st.append(vs[i+1])
            edge[vs[i+1]] = []

        for i in range(len(st) - 1):
            edge[st[i]].append(st[i+1])
            par[st[i+1]] = st[i]

        par[st[0]] = -1
        return st[0], edge, par

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
        val = c
        for x,d in (a, depth[a] - depth[c]), (b, depth[b] - depth[c]):
            for i in range(d.bit_length()):
                if (d >> i) & 1:
                    val = f(val, data[i][x])
                    x = parent[i][x]
        return val

def solve():
    n,m = LII()

    dnc = HLPP(n+1)

    for _ in range(m):
        u,v,w = LII()
        dnc.add_edge(u,v,w)
    
    print(dnc.maxflow(1,n))

    #L1 = LII()
    #st = SI()
solve()