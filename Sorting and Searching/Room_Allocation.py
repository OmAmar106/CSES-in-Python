import sys,math,random
from heapq import heappush,heappop,heapify
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

def solve():
    n = int(sys.stdin.readline().strip())
    L1 = []
    for i in range(n):
        L = list(map(int, sys.stdin.readline().split()))
        L1.append((L[0],L[1],i))
    L1.sort()
 
    ans = [0]*n
    
    H = []
    cur = 1
 
    for i in range(n):
        if H and L1[i][0]>H[0][0]:
            ans[L1[i][2]] = H[0][1]
            heappop(H)
        else:
            ans[L1[i][2]] = cur
            cur += 1 
        heappush(H,(L1[i][1],ans[L1[i][2]]))
    print(cur-1)
    print(*ans)
    
solve()