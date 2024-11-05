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
#Combinatorics : pnc, Diophantine Equations : dpheq
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

class SegmentTree:
    """
        Remember to change the func content as well as the initializer to display the content
    """
    @staticmethod
    def func(a, b):
        # Change this function depending upon needs
        return max(a, b)
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        self.build(data)
    def build(self, data):
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.func(self.tree[i * 2], self.tree[i * 2 + 1])
    def update(self, pos, value):
        # Update the value at the leaf node
        if self.tree[pos+self.n]>=value:
            return
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
        max_val = float('-inf')
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
    n = int(sys.stdin.readline().strip())
    L2 = []
    L3 = []
    L4 = []
    L5 = []
    for i in range(n):
        L = list(map(int, sys.stdin.readline().split()))
        L2.append((L[0],L[1],L[2]))

    # L3.sort()
    # count = 1
    # d = {}
    # for i in range(len(L3)):
    #     if L3[i] in d:
    #         continue
    #     d[L3[i]] = count
    #     count += 1
    
    L2 = sorted(L2,key=lambda x:x[1])
    for i in range(len(L2)):
        L3.append(L2[i][0])
        L4.append(L2[i][1])
        L5.append(L2[i][2])
    # seg = SegmentTree([0 for i in range(count+1)])
    
    # print(L2)
    dp = [L5[0]]

    for i in range(1,len(L2)):
        # print(i,seg.query(0,d[L2[i][0]])+L2[i][2])
        # k = seg.query(0,d[L2[i][0]])+L2[i][2]
        # seg.update(d[L2[i][1]],k)
        # ans = max(ans,k)
        dp.append(dp[i-1])
        pos = bisect_right(L4,L3[i]-1)
        if pos>0:
            dp[i] = max(dp[i],dp[pos-1]+L5[i])
        else:
            dp[i] = max(dp[i],L5[i])

    print(dp[-1])
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()