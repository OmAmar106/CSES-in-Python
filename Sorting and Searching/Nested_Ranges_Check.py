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

def solve():
    n = int(sys.stdin.readline().strip())
    L2 = []
    for i in range(n):
        L = list(map(int, sys.stdin.readline().split()))
        L2.append([L[0],L[1],i])
    L2 = sorted(L2,key = lambda x:(x[0],-x[1]))
    ans = [0]*n
    mini = float('inf')
    for i in L2[::-1]:
        if i[1]>=mini:
            ans[i[2]] = 1
        mini = min(mini,i[1])
    print(*ans)
    ans = [0]*n
    maxi = 0
    for i in L2:
        if i[1]<=maxi:
            ans[i[2]] = 1
        maxi = max(maxi,i[1])
    print(*ans)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()