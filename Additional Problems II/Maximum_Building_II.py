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

def solve():
    n,m = LII()

    pref = [0]*m
    ans = [[0]*m for i in range(n)]

    for i in range(n):
        st = SI()
        for j in range(m):
            if st[j]=='.':
                pref[j] += 1
            else:
                pref[j] = 0

        # for j in range(len(pref)):
        #     f = float('inf')
        #     for k in range(j,-1,-1):
        #         f = min(f,pref[k])
        #         if pref[k]==0:
        #             break
        #         # ans[f-1][(j-k)] += 1
        #         ans[f-1][(j-k)] += 1
        st = []
        ans1 = []
        for j in range(len(pref)):
            while st and pref[j]<=pref[st[-1]]:
                st.pop()
            if st:
                ans1.append(st[-1])
            else:
                ans1.append(-1)
            st.append(j)

        st = []
        ans2 = []
        for j in range(len(pref)-1,-1,-1):
            while st and pref[j]<pref[st[-1]]:
                st.pop()
            if st:
                ans2.append(st[-1])
            else:
                ans2.append(len(pref))
            st.append(j)

        ans2.reverse()
        # print(pref)
        # print(ans1)
        # print(ans2)

        for j in range(len(pref)):
            if pref[j]==0:
                continue
            # from 0 to ans2[j]-j tak +1 krna hain 
            # for k in range(j,ans2[j]):
            #     if (k-j)+(j-ans1[j])<len(ans[pref[j]-1]):
            #         ans[pref[j]-1][(k-j)+(j-ans1[j])] -= 1
            #     ans[pref[j]-1][k-j] += 1

            # prefix sum of prefix sum banaya
            ans[pref[j]-1][0] += 1
            if ans2[j]-j<m:
                ans[pref[j]-1][ans2[j]-j] -= 1
            if j-ans1[j]<m:
                ans[pref[j]-1][j-ans1[j]] -= 1
            if ans2[j]-ans1[j]<m:
                ans[pref[j]-1][ans2[j]-ans1[j]] += 1

    for k in range(2):
        for j in range(m-1):
            ans[-1][j+1] += ans[-1][j]

    for i in range(n-2,-1,-1):
        for k in range(2):
            for j in range(m-1):
                ans[i][j+1] += ans[i][j]
        for j in range(m):
            ans[i][j] += ans[i+1][j]

    for i in ans:
        print(*i)

    #L1 = LII()
    #st = SI()
solve()