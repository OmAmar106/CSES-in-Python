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
#Graph1(axtree,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, DynamicConnectivity: odc
#Treap: treap, MaxMatching(Kuhn,Hopcroft): graphmatch
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

class Matrix:
    MOD = 10**9+7
    @staticmethod
    def transpose(mat):
        return [list(c) for c in zip(*mat)]
    def __init__(self,M):
        self.M=M
    def minmul(self,o):
        L,B=self.M,o.M
        r=len(L); c=len(B[0])
        return Matrix([[min(L[i][k]+B[k][j] for k in range(len(B)))%self.MOD for j in range(c)] for i in range(r)])
    def __matmul__(self,o):
        L,B=self.M,o.M
        r=len(L); c=len(B[0])
        return Matrix([[sum(L[i][k]*B[k][j] for k in range(len(B)))%self.MOD for j in range(c)] for i in range(r)])
    def __pow__(self,p):
        M=self; n=len(M.M)
        R=Matrix([[i==j for j in range(n)] for i in range(n)])
        while p:
            R = R@M if p&1 else R
            M = M@M
            p//=2
        return R
    @staticmethod
    def gauss(A,mod=10**9+7):
        m,n=len(A),len(A[0])-1; r=0; L=[-1]*n
        for c in range(n):
            for i in range(r,m):
                if A[i][c]: A[r],A[i]=A[i],A[r]
                break
            else:
                continue
            k=pow(A[r][c],-1,mod)
            for j in range(c,n+1):
                A[r][j]=A[r][j]*k%mod
            for i in range(m):
                if i!=r and A[i][c]:
                    f=A[i][c]
                    for j in range(c,n+1):
                        A[i][j]=(A[i][j]-f*A[r][j])%mod
            L[c]=r
            r+=1
        if any(A[i][n] for i in range(r,m)):
            return None
        return [A[L[i]][n] if L[i]!=-1 else 0 for i in range(n)]
    def __str__(self):
        return '\n'.join(' '.join(map(str,row)) for row in self.M)
    def __repr__(self):
        return f"Matrix({self.M})"
    def __iter__(self):
        return iter(self.M)

def solve():
    k = II()

    d = [[0]*64 for i in range(64)]

    def ispossible(x,y):
        return 0<=x<8 and 0<=y<8

    for i in range(8):
        for j in range(8):
            count = 0
            for dx,dy in dirs:
                if ispossible(i+dx,j+dy):
                    count += 1
            for dx,dy in dirs:
                if ispossible(i+dx,j+dy):
                    d[i*8+j][(i+dx)*8+(j+dy)] += 1/count

    d = Matrix(d)**k
    # dp = [[0]*8 for i in range(8)]

    # for j in range(64):
    #     for k in range(64):
    #         # start at j, ended at k
    #         dp[k//8][k%8] += d.M[j][k]
    dp = [[1]*8 for _ in range(8)]

    for start in range(64):
        for end in range(64):
            x,y = end//8,end%8
            dp[x][y] *= (1-d.M[start][end])

    ans = str(round(sum(sum(i) for i in dp),6))
    # print(round(sum(sum(i) for i in dp),6))
    while len(ans)-ans.index('.')!=7:
        ans += '0'
    print(ans)
    #L1 = LII()
    #st = SI()
solve()