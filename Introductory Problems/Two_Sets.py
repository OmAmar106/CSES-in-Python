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
    if n%4==0:
        print("YES")
        ans = []
        ans1 = []
        for i in range(0,n,4):
            ans.append(i+1)
            ans.append(i+4)
            ans1.append(i+2)
            ans1.append(i+3)
        print(len(ans))
        print(*ans)
        print(len(ans1))
        print(*ans1)
    elif n%4==3:
        print("YES")
        ans = [1,2]
        ans1 = [3]
        for i in range(0,n-3,4):
            ans.append(4+i)
            ans.append(4+i+3)
            ans1.append(4+i+1)
            ans1.append(4+i+2)
        print(len(ans))
        print(*ans)
        print(len(ans1))
        print(*ans1)
    else:
        print("NO")
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()