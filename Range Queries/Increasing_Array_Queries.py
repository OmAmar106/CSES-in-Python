import sys,math,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(2**62)
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

#String hashing : sh, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

class SegmentTree:
    """
        Remember to change the func content as well as the initializer to display the content
    """
    @staticmethod
    def func(a, b):
        # Change this function depending upon needs
        if a[0]>b[0]:
            return a
        else:
            return b
    def __init__(self, data):
        self.n = len(data)
        self.tree = [(-1,-1)] * (2 * self.n)
        self.build(data)
    def build(self, data):
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.func(self.tree[i * 2], self.tree[i * 2 + 1])
    def update(self, pos, value):
        # Update the value at the leaf node
        pos += self.n
        # For updating
        self.tree[pos] = value
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
        max_val = (float('-inf'),-1)
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
    
def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    st = []
    suff = [0 for i in range(n)]
    for i in range(n-1,-1,-1):
        while st and L[st[-1]]<=L[i]:
            st.pop()    
        if not st:
            suff[i] = L[i]*(n-i)
        else:
            suff[i] = suff[st[-1]]+L[i]*(st[-1]-i)
        st.append(i)
    pref = [0]
    for i in L:
        pref.append(pref[-1]+i)
    seg = SegmentTree(list(zip(L,[i for i in range(n)])))
    for i in range(q):
        L1 = list(map(int, sys.stdin.readline().split()))
        f = seg.query(L1[0]-1,L1[1])
        k = suff[L1[0]-1]-suff[f[1]]
        k += f[0]*(L1[1]-f[1])
        print(k-pref[L1[1]]+pref[L1[0]-1])
    #st = sys.stdin.readline().strip()
solve()