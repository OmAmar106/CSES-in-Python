import sys,math,cmath,random,os
# from heapq import heappush,heappop
# from bisect import bisect_right,bisect_left
# from collections import Counter,deque,defaultdict
# from itertools import permutations,combinations
from io import BytesIO, IOBase
# from decimal import Decimal,getcontext

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
# MOD = 10**9 + 7
# RANDOM = random.randrange(1,2**62)
# def gcd(a,b):
#     while b:
#         a,b = b,a%b
#     return a
# def lcm(a,b):
#     return a//gcd(a,b)*b
# def w(x):
#     return x ^ RANDOM
# II = lambda : int(sys.stdin.readline().strip())
# LII = lambda : list(map(int, sys.stdin.readline().split()))
# MI = lambda x : x(map(int, sys.stdin.readline().split()))
SI = lambda : sys.stdin.readline().strip()
# SLI = lambda : list(map(lambda x:ord(x)-97,sys.stdin.readline().strip()))
# LII_1 = lambda : list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
# LII_C = lambda x : list(map(x, sys.stdin.readline().split()))
# MATI = lambda x : [list(map(int, sys.stdin.readline().split())) for _ in range(x)]
##

#String hashing: shclass, fenwick sortedlist: fsortl, Number: numtheory/numrare, SparseTable: SparseTable
#Bucket Sorted list: bsortl, Segment Tree(lp,selfop): SegmentTree, bootstrap: bootstrap, Trie: tries
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, BitArray: bitarray
#Combinatorics: pnc, Diophantine Equations: dpheq, DSU: DSU, Geometry: Geometry, FFT: fft, XOR_dict: xdict
#Persistent Segment Tree: perseg, Binary Trie: b_trie, HLD: hld, String funcs: sf, Segment Tree(lp): SegmentOther
#Graph1(dnc,bl): graphadv, Graph2(khn,sat): 2sat, Graph3(fltn,bprt): graphflatten, Graph4(ep,tp,fw,bmf): graphoth
#Graph5(djik,bfs,dfs): graph, Graph6(dfsin): dfsin, utils: utils, Persistent DSU: perdsu, Merge Sort Tree: sorttree
#2-D BIT: 2DBIT, MonoDeque: mono
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

def solve():
    st = SI()
    dirs = ((0,1),(0,-1),(1,0),(-1,0))
    n = 7
    visited = [[False]*n for i in range(n)]
 
    f = {'D':(1,0),'U':(-1,0),'R':(0,1),'L':(0,-1)}
 
    def rec(i,x,y):
        if i==len(st):
            return 1*(x==6 and y==0)
        if visited[x][y] or (x==6 and y==0):
            return 0
        vis1 = [True]*4
        for count,(dx,dy) in enumerate(dirs):
            if 0<=x+dx<7 and 0<=y+dy<7:
                vis1[count] = visited[x+dx][y+dy]
 
        if not vis1[2] and not vis1[3] and vis1[0] and vis1[1]:
            return 0
        elif not vis1[0] and not vis1[1] and vis1[2] and vis1[3]:
            return 0
        elif 0<=x-1<7 and 0<=y+1<7 and visited[x-1][y+1] and not vis1[0] and not vis1[3]:
            return 0
        elif 0<=(x+1)<7 and 0<=(y+1)<7 and visited[x+1][y+1] and not vis1[0] and not vis1[2]:
            return 0
        elif 0<=(x-1)<7 and 0<=(y-1)<7 and visited[x-1][y-1] and not vis1[1] and not vis1[3]:           
            return 0
        elif 0<=(x+1)<7 and 0<=(y-1)<7 and visited[x+1][y-1] and not vis1[1] and not vis1[2]:
            return 0
        
        visited[x][y] = True
        ans = 0
        if st[i]=='?':
            for dx,dy in dirs:
                if 0<=(x+dx)<7 and 0<=(y+dy)<7:
                    ans += rec(i+1,x+dx,y+dy)
        else:
            dx = f[st[i]][0];dy = f[st[i]][1]
            if 0<=x+dx<7 and 0<=y+dy<7:
                ans += rec(i+1,x+dx,y+dy)
        visited[x][y] = False
        return ans
 
    print(rec(0,0,0))
    #L1 = LII()
    #st = SI()
solve()