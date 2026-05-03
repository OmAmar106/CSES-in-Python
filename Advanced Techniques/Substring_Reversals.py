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
 
class Treap:
    class Node:
        __slots__ = ('val','prio','left','right','size','rev')
        def __init__(self,val):
            self.val=val
            self.prio=random.random()
            self.left=None
            self.right=None
            self.size=1
            self.rev=False
 
    def __init__(self,s=""):
        """Build treap from string s in O(n) by repeated merges"""
        self.root=None
        for ch in s:
            node=self.Node(ch)
            self.root=self._merge(self.root,node)
 
    def _update(self,node):
        node.size=1
        if node.left: node.size+=node.left.size
        if node.right: node.size+=node.right.size
 
    def _push(self,node):
        if node and node.rev:
            node.rev=False
            node.left, node.right = node.right, node.left
            if node.left: node.left.rev=not node.left.rev
            if node.right: node.right.rev=not node.right.rev
 
    def _split(self,node,count):
        """Iterative split: first count into left"""
        if not node: return (None,None)
        self._push(node)
        left_size = node.left.size if node.left else 0
        if count<=left_size:
            # split left subtree
            l,l2=self._split(node.left,count)
            node.left=l2
            self._update(node)
            return (l,node)
        else:
            # split right subtree
            r1,r=self._split(node.right,count-left_size-1)
            node.right=r1
            self._update(node)
            return (node,r)
 
    def _merge(self,a,b):
        """Iterative merge using loop"""
        if not a or not b: return a or b
        # choose root
        if a.prio>b.prio:
            root=a
            self._push(root)
            root.right=self._merge(root.right,b)
        else:
            root=b
            self._push(root)
            root.left=self._merge(a,root.left)
        self._update(root)
        return root
 
    def reverse(self,l,r):
        """Reverse substring [l,r) 0-indexed"""
        A,B=self._split(self.root,l)
        M,C=self._split(B,r-l)
        if M: M.rev=not M.rev
        self.root=self._merge(self._merge(A,M),C)
 
    def __repr__(self):
        """Iterative inorder traversal"""
        res=[]
        stack=[]
        node=self.root
        while stack or node:
            while node:
                self._push(node)
                stack.append(node)
                node=node.left
            node=stack.pop()
            res.append(node.val)
            node=node.right
        return ''.join(res)
def solve():
    n,m = list(map(int, sys.stdin.readline().split()))
    #L1 = list(map(int, sys.stdin.readline().split()))
    st = sys.stdin.readline().strip()
    tp = Treap(st)
    for i in range(m):
        l,r = list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
        tp.reverse(l,r+1)
    print(tp)        
solve()
