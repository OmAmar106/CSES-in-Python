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
#Bucket Sorted list: bsortl, bootstrap: bootstrap, Trie: tries, Segment Tree(lp): SegmentOther, Treap: treap
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, DynamicConnectivity: odc
#Graph1(axtree,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono, matrix: nummat, SuffixAutomaton: sautomaton, linalg: linalg, SquareRtDecomp: sqrt
#Grapth7(bridges): graph_dmgt, FWHT(^,|,&): fwht, Graph8(centr_decom): graph_decom, DpOptimize(knth,dnc): dpopt
#2D-SegmentTree: seg2d, Rollback/Par DSU: rbdsu, MaxFlow(Dnc,HLPP): graphflow, MaxMatching(Kuhn,Hopcroft): graphmatch
#Segment Tree(Node): SegmentNode
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
#if os.environ.get('LOCAL'):sys.stdin = open(r'input.txt', 'r');sys.stdout = sys.stderr = open(r'output.txt','w')
#if os.environ.get('LOCAL'):import hashlib;print('Hash Value :',hashlib.md5(open(__file__, 'rb').read()).hexdigest());

def solve():
    n = II()
 
    # def f(n):
    #     n = str(n)
    #     if len(n)==1:
    #         return 1
    #     f = len(n)-2
    #     ans = (10**f)*(f+1)
    #     # if int(n)<=12:
    #     #     print(n,ans)
    #     if n[0]!='1':
    #         ans += 10**(len(n)-1)
    #     elif len(n)!=1:
    #         ans += int(n[1:])+1
    #     # if int(n)<=10:
    #     #     print(n,ans)
    #     for i in range(1,len(n)):
    #         # idhar kitne par 1 hoga
    #         ans += (int(n[:i])-1)*(10**len(n)-2)
    #         if int(n[i])>=1:
    #             if i!=len(n)-1:
    #                 ans += int(n[i+1:])+1
    #             else:
    #                 ans += 1
    #     return ans
    def f(n):
        
        # for j in range(102):
        #     if '1' in str(j):
        #         print(j)

        n = list(map(int,str(n)))

        # print(n)

        dp = [[[0,0,0] for _ in range(10)] for i in range(len(n))]

        n = n[::-1]

        for i in range(10):
            if i==1:
                if i<n[0]:
                    dp[0][i][0] = 1
                elif i==n[0]:
                    dp[0][i][1] = 1
                else:
                    dp[0][i][2] = 1
        
        for i in range(1,len(n)):
            for j in range(10):
                for k in range(10):
                    if j<n[i]:
                        dp[i][j][0] += dp[i-1][k][0]
                        dp[i][j][0] += dp[i-1][k][1]
                        dp[i][j][0] += dp[i-1][k][2]
                    elif j==n[i] and k<=n[i-1]:
                        dp[i][j][0] += dp[i-1][k][0]
                        dp[i][j][2] += dp[i-1][k][2]
                        if k==n[i-1]:
                            dp[i][j][1] += dp[i-1][k][1]
                    else:
                        dp[i][j][2] += dp[i-1][k][2]+dp[i-1][k][0]+dp[i-1][k][1]
                
                if j==1:
                    # print(dp[i][j][0])
                    if j==n[i]:
                        dp[i][j][2] += pow(10,i)-1-int(''.join(map(str,n[:i][::-1])))
                        dp[i][j][1] += 1
                        dp[i][j][0] += int(''.join(map(str,n[:i][::-1])))
                    if j>n[i]:
                        dp[i][j][2] += pow(10,i)
                    if j<n[i]:
                        dp[i][j][0] += pow(10,i)
                        # dp[i][j][2] += int('9'*i)-int(''.join(map(str,n[:i])))
        return sum(j[0]+j[1] for j in dp[-1])

    # print(rec(1,1,[3,1]))
 
    start = 0
    end = 10**18
    while start<=end:
        mid = (start+end)//2
        if f(mid)<=n:
            start = mid+1
        else:
            end = mid-1

    print(start-1)

solve()