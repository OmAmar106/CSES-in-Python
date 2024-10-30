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
    L2 = []
    L3 = []
    for i in range(int(sys.stdin.readline().strip())):
        L = list(map(int, sys.stdin.readline().split()))
        L2.append(L[0])
        L3.append(L[1])
    L2.sort()
    L3.sort()
    j = 0
    i = 0   
    fans = 0
    ans = 0
    while i<len(L2) and j<len(L3):
        if L2[i]<L3[j]:
            ans += 1
            i += 1
        else:
            ans -= 1
            j += 1
        fans = max(fans,ans)
    print(fans)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()