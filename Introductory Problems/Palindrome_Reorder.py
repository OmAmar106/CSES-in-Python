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
    #L1 = list(map(int, sys.stdin.readline().split()))
    st = sys.stdin.readline().strip()
    d = Counter(st)
    ans = [0 for i in range(len(st))]
    end = len(st)-1
    start = 0
    for i in d:
        while d[i]>=2:
            d[i] -= 2
            ans[end] = i
            ans[start] = i
            end -= 1
            start += 1
        if d[i]:
            d[i] -= 1
            ans[len(ans)//2]=i
    try:
        print(''.join(ans))
    except:
        print("NO SOLUTION")
solve()