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
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

def solve():
    n,k = list(map(int, sys.stdin.readline().split()))
    def rec(n,k):
        if n==1:
            return 1
        if k<=((n+1)//2):
            if 2*k>n:
                return (2*k)%n
            return 2*k
        temp = rec(n//2,k-(n+1)//2)
        if n%2==0:
            return 2*temp-1
        else:
            return 2*temp+1
    print(rec(n,k))

    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
for _ in range(int(sys.stdin.readline().strip())):
    solve()