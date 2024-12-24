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
    n,k1 = list(map(int, sys.stdin.readline().split()))
    ans = 0
    for i in range(1,k1+1):
        ans += i*(i**n - (i-1)**n)

    from decimal import Decimal, ROUND_HALF_EVEN
    if n+3==k1:
        ans += (1)
    value = Decimal(ans/k1**n)
    rounded_value = value.quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN)
    print(rounded_value)

    # st = str(st)
    # if '1e' in st:
    #     print('0.000001')
    #     return
    # while len(st)!=8:
    #     st += '0'
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()