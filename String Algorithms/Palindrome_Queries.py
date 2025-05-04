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
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie: Tries
#Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
#Persistent Segment Tree: perseg, FreqGraphs: bgraph, Binary Trie: b_trie, XOR_dict: xdict, HLD: hld
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

class SegmentTree:
    """
        Remember to change the func content as well as the initializer to display the content
    """
    @staticmethod
    def func(a, b):
        # Change this function depending upon needs
        return (a+b)%MOD
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        self.pow = [1]
        for i in range((self.n)):
            self.pow.append((self.pow[-1]*137)%MOD)
        self.build(data)
    def build(self, data):
        for i in range(self.n):
            self.tree[self.n + i] = (ord(data[i])*(self.pow[i]))%MOD
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.func(self.tree[i * 2], self.tree[i * 2 + 1])
    def update(self, pos, value):
        # Update the value at the leaf node
        pos += self.n
        # For updating
        self.tree[pos] = (ord(value)*(self.pow[pos-self.n]))%MOD
        # self.tree[pos] += value
        # If you want to add rather than update
        while pos > 1:
            pos //= 2
            self.tree[pos] = self.func(self.tree[2 * pos], self.tree[2 * pos + 1])
    def query(self, left, right):
        # Query the maximum value in the range [left, right)
        left += self.n
        right += self.n
        # Change the initializer depending upon the self.func
        max_val = 0
        ##
        while left < right:
            if left % 2:
                max_val = self.func(max_val, self.tree[left])
                left += 1
            if right % 2:
                right -= 1
                max_val = self.func(max_val, self.tree[right])
            left //= 2
            right //= 2
        return max_val
    def __repr__(self):
        print('Seg[',end='')
        for i in range(self.n):
            if i!=self.n-1:
                print(self.query(i,i+1),end=', ')
            else:
                print(self.query(i,i+1),end=']')
        print()

def solve():
    n,m = list(map(int, sys.stdin.readline().split()))
    #L1 = list(map(int, sys.stdin.readline().split()))
    st = sys.stdin.readline().strip()
    s1 = SegmentTree(st)
    s2 = SegmentTree(st[::-1])
    for i in range(m):
        k,x,y = list(sys.stdin.readline().split())
        if k=='1':
            s1.update(int(x)-1,y)
            s2.update(n-int(x),y)
        else:
            f = s1.query(int(x)-1,int(y)-1)
            f1 = s2.query(n-int(y),n-int(x))
            if int(x)-1>n-int(y):
                f1 *= s2.pow[int(x)-1-(n-int(y))]
            else:
                f *= s2.pow[n-int(y)-(int(x)-1)]
            f %= MOD
            f1 %= MOD
            if f==f1:
                print("YES")
            else:
                print("NO")
solve()
