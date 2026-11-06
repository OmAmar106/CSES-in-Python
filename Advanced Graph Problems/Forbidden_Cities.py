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

#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare/numfast, SparseTable: SparseTable
#Bucket sortedlist: bsortl, bootstrap: bootstrap, Trie: tries, Segment Tree(lp): SegmentOther, Treap: treap
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, DynamicConnectivity: odc
#Graph1(axtree,bl): graphadv, Graph2(khn,sat,scc): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, Suffix/KMPAutomaton: sautomaton, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, MaxMatching(Kuhn,Hopcroft): graphmatch
#Segment Tree(Node): SegmentNode, mcmf: mcmf, pref2d: pref2d, RecSegTree: SegmentTreeRec, AVL sortedlist: asortl
#bwt: bwt, linalg: linalg, LCT: lct
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

dirs8 = [
    (0, 1),	# 0: up
    (1, 1),	# 1: up-right
    (1, 0),	# 2: right
    (1, -1),   # 3: down-right
    (0, -1),   # 4: down
    (-1, -1),  # 5: down-left
    (-1, 0),   # 6: left
    (-1, 1)	# 7: up-left
]

dir_id = {
    (0, 1): 0,
    (1, 1): 1,
    (1, 0): 2,
    (1, -1): 3,
    (0, -1): 4,
    (-1, -1): 5,
    (-1, 0): 6,
    (-1, 1): 7
}

def rotate_clockwise_45(ind):
    return (ind+1)%8

def rotate_anti_clockwise_45(ind):
    return (ind-1)%8

def reverse(ind):
    return (ind+4)%8

# slash mirror '/'
#
# Light reflecting on:
#
#	 /
#
# slash[d] gives new direction after hitting '/'
slash = [2, 1, 0, 7, 6, 5, 4, 3]
# backslash[d] gives new direction after hitting ''
backslash = [6, 5, 4, 3, 2, 1, 0, 7]
# vertical_mirror[d] gives new direction after hitting '|'
vertical_mirror = [4, 3, 2, 1, 0, 7, 6, 5]
horizontal_mirror = [4, 5, 6, 7, 0, 1, 2, 3]

def extras():
    # use Decimal(number) for context to work
    # slows down the code significantly
    # by default, there is precision of 10-6
    getcontext().prec = 50
    # more precision, use Fraction
    from fractions import Fraction
    sys.setrecursionlimit(10**6)
    sys.set_int_max_str_digits(10**5)
# extras()

def interactive():
    import builtins
    # print(globals())
    globals()['print'] = lambda *args, **kwargs: builtins.print(*args, flush=True, **kwargs)
# interactive()

def reduce_time():
    # reduces run time, but increases the chance of mle
    import gc
    gc.disable()
# reduce_time()

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


def solve():
    n,m,q = LII()

    d = GI(n,m)

    tin = [-1]*n
    tout = [-1]*n
    low = [float('inf')]*n
    parent = [-1]*n

    time = 0

    # def dfs(cur,prev):
    #     nonlocal time
    #     tin[cur] = time
    #     time += 1
    #     low[cur] = tin[cur]
    #     for i in d[cur]:
    #         if i!=prev:
    #             if tin[i]==-1:
    #                 parent[i] = cur
    #                 dfs(i,cur)
    #                 low[cur] = min(low[cur],low[i])
    #             else:
    #                 low[cur] = min(low[cur],tin[i])
    #     tout[cur] = time
    #     time += 1

    gr = d
    n = len(gr)
    visited = [False]*n
    for root in range(n):
        if visited[root]:
            continue
        parent[root] = -1
        stack = [(root,0)]
        while stack:
            u,idx = stack[-1]
            if not visited[u]:
                visited[u] = True
                tin[u] = low[u] = time
                time += 1
            if idx<len(gr[u]):
                v = gr[u][idx]
                stack[-1] = (u,idx+1)
                if v==parent[u]:
                    continue
                if not visited[v]:
                    parent[v] = u
                    stack.append((v,0))
                else:
                    low[u] = min(low[u],tin[v])
            else:
                tout[u] = time
                time += 1
                stack.pop()
                if parent[u]!=-1:
                    low[parent[u]] = min(low[parent[u]],low[u])

    # for i in range(n):
    #     if tin[i]==-1:
    #         dfs(i,-1)

    def f(elem,subtree):
        return tin[subtree]<=tin[elem]<=tout[subtree]

    for _ in range(q):
        a,b,c = LII_1()
        if b==c or a==c:
            print("NO")
            continue

        comp = []
        for v in d[c]:
            if c==parent[v]:
                if tin[c]<=low[v]:
                    comp.append(v)

        def get(x):
            for v in comp:
                if f(x,v):
                    return v
            return -1

        if get(a)!=get(b):
            print("NO")
        else:
            print("YES")

        # this O(q*deg) complexity but passes the tcs
        # O(q) would be solving for all same c's together
        
    #L1 = LII()
    #st = SI()
solve()