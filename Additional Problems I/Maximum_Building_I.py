import sys,math,cmath,random,os
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations
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

# functions #
# MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(1,2**62)
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
##

#String hashing: sh/shclass, fenwick sortedlist: fsortl, Number: numtheory, SparseTable: SparseTable
#Bucket Sorted list: bsortl, Segment Tree(lazy propogation): SegmentTree/Other, bootstrap: bootstrap
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
#Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
#Persistent Segment Tree: perseg, FreqGraphs: bgraph, Binary Trie: b_trie, XOR_dict: xdict, HLD: hld
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

def solve():
    n,m = list(map(int, sys.stdin.readline().split()))
    L = []
    ans = 0
    count = [0]*m
    for i in range(n):
        st = sys.stdin.readline().strip()
        for j in range(len(st)):
            if st[j]=='*':
                count[j] = 0
            else:
                count[j] += 1
        
        st = []
        prevlarge = [-1]*m

        for j in range(m):
            while st and count[st[-1]]>=count[j]:
                st.pop()
            if st:
                prevlarge[j] = st[-1]
            st.append(j)
        
        st = []

        for j in range(m-1,-1,-1):
            while st and count[st[-1]]>=count[j]:
                st.pop()
            if st:
                ans = max(ans,(st[-1]-prevlarge[j]-1)*count[j])
            else:
                ans = max(ans,(m-prevlarge[j]-1)*count[j])
            st.append(j)

        
    print(ans)

solve()