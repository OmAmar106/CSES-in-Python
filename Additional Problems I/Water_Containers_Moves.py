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
#Graph5(djik,H,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, nummat: matrix, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

def solve():
    a,b,x3 = LII()

    dist = [[float('inf')]*(b+1) for i in range(a+1)]
    parent = [[-1]*(b+1) for i in range(a+1)]
    type = [[-1]*(b+1) for i in range(a+1)]
    dist[0][0] = 0
    H = [(0,0,0)]

    while H:
        d,x,y = heappop(H)
        if d==dist[x][y]:
            if x:
                if dist[0][y]>dist[x][y]+x:
                    dist[0][y] = dist[x][y]+x
                    heappush(H,(d+x,0,y))
                    parent[0][y] = (x,y)
                    type[0][y] = 3
            if y:
                if dist[x][0]>dist[x][y]+y:
                    dist[x][0] = dist[x][y]+y
                    heappush(H,(d+y,x,0))
                    parent[x][0] = (x,y)
                    type[x][0] = 4
            if dist[a][y]>dist[x][y]+(a-x):
                dist[a][y] = dist[x][y]+(a-x)
                heappush(H,(d+(a-x),a,y))
                parent[a][y] = (x,y)
                type[a][y] = 1
            if dist[x][b]>dist[x][y]+(b-y):
                dist[x][b] = dist[x][y]+(b-y)
                heappush(H,(d+(b-y),x,b))
                parent[x][b] = (x,y)
                type[x][b] = 2
            t = min(x,b-y)
            if dist[x-t][y+t]>dist[x][y]+t:
                dist[x-t][y+t] = dist[x][y]+t
                heappush(H,(d+t,x-t,y+t))
                parent[x-t][y+t] = (x,y)
                type[x-t][y+t] = 5
            t = min(y,a-x)
            if dist[x+t][y-t]>dist[x][y]+t:
                dist[x+t][y-t] = dist[x][y]+t
                heappush(H,(d+t,x+t,y-t))
                parent[x+t][y-t] = (x,y)
                type[x+t][y-t] = 6

    mini = float('inf')
    ans = -1

    x = x3
    if x<=a:
        for i in range(b+1):
            if dist[x][i]<mini:
                mini = dist[x][i]
                ans = (x,i)
    
    if ans==-1:
        print(-1)
        return
    
    fans = []
    while ans!=-1:
        fans.append(type[ans[0]][ans[1]])
        ans = parent[ans[0]][ans[1]]
    
    fans.reverse()
    a1 = 0
    b1 = 0
    ans = 0
    for i in fans:
        if i==1:
            ans += (a-a1)
            a1 = a
        elif i==2:
            ans += (b-b1)
            b1 = b
        elif i==3:
            ans += a1
            a1 = 0
        elif i==4:
            ans += b1
            b1 = 0
        elif i==5:
            t = min(a1,b-b1)
            b1 += t
            a1 -= t
            ans += t
        elif i==6:
            t = min(a-a1,b1)
            b1 -= t
            a1 += t
            ans += t

    print(len(fans)-1,ans)
    for i in fans:
        if i==1:
            print("FILL A")
        elif i==2:
            print("FILL B")
        elif i==3:
            print("EMPTY A")
        elif i==4:
            print("EMPTY B")
        elif i==5:
            print("MOVE A B")
        elif i==6:
            print("MOVE B A")

    #L1 = LII()
    #st = SI()
solve()