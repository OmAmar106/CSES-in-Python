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

def shallowest_decomposition_tree(graph, root=0):
    # returns a root and a directed tree, iterate on it and you
    # will find the new centroid
    # slightly faster than centroid_decomposition_tree
    ctz = lambda x:(x & -x).bit_length()-1
    n = len(graph)
    forbidden = [0] * n
    decomposition_tree = [[] for _ in range(n)]
    stacks = [[] for _ in range(n.bit_length())]
    def extract_chain(labels, u):
        while labels:
            label = labels.bit_length() - 1
            labels ^= 2**label
            v = stacks[label].pop()
            decomposition_tree[u].append(v)
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
            forbidden[u] = forbidden_once | (2**forbidden_twice.bit_length() - 1)
            label_u = ctz(forbidden[u] + 1)
            stacks[label_u].append(u)
            for v in graph[u]:
                extract_chain((forbidden[v] + 1) & (2**label_u - 1), u)
    max_label = (forbidden[root] + 1).bit_length() - 1
    decomposition_root = stacks[max_label].pop()
    extract_chain((forbidden[root] + 1) & (2**max_label - 1), decomposition_root)
    return decomposition_tree, decomposition_root

def centroid_decomposition_tree(graph):
    # returns a root and a directed tree, iterate on it and you
    # will find the new centroid
    n = len(graph)
    graph = [c[:] for c in graph]
    bfs = [0]
    for node in bfs:
        bfs += graph[node]
        for nei in graph[node]:
            graph[nei].remove(node)
    size = [0] * n
    for node in reversed(bfs):
        size[node] = 1 + sum(size[child] for child in graph[node])
    decomposition_tree = [[] for _ in range(n)]
    def centroid_reroot(u):
        N = size[u]
        while True:
            for v in graph[u]:
                if size[v] > N // 2:
                    size[u] = N - size[v]
                    graph[u].remove(v)
                    graph[v].append(u)
                    u = v
                    break
            else:
                decomposition_tree[u] = [centroid_reroot(v) for v in graph[u]]
                return u
    decomposition_root = centroid_reroot(0)
    return decomposition_tree, decomposition_root

def solve():
    n,k = LII()
    d = GI(n)
    L,root = shallowest_decomposition_tree(d)

    gr = L

    n = len(gr)
    stack = [root]
    
    dead = [False]*n
    
    ans = 0
    
    while stack:
        start = stack.pop()
        count = Counter()
        count[0] = 1
        dead[start] = True
        for j in gr[start]:
            stack.append(j)

        for j in d[start]:
            if dead[j]:
                continue

            depth = Counter()
            visited = set()
            st = [j]
            depth[j] = 1
            while st:
                start1 = st.pop()
                ans += count[k-depth[start1]]
                visited.add(start1)
                if depth[start1]==k:
                    continue
                for child in d[start1]:
                    if child not in visited and not dead[child]:
                        depth[child] = depth[start1]+1
                        st.append(child)

            visited = set()
            st = [j]
            while st:
                start1 = st.pop()
                count[depth[start1]] += 1
                visited.add(start1)
                if depth[start1]==k:
                    continue
                for child in d[start1]:
                    if child not in visited and not dead[child]:
                        st.append(child)

    print(ans)

    #L1 = LII()
    #st = SI()
solve()