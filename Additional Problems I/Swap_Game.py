import sys,math,cmath,random
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

#String hashing : sh/shclass, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull, Trie/Treap : Tries
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU, Geometry: Geometry, FFT: fft
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

def solve():
    k3 = 0
    for i in range(3):
        L = (list(map(int, sys.stdin.readline().split())))
        for j in range(len(L)):
            k3 *= 10
            k3 += L[j]
    
    d = {}
    def func(k):
        if k not in d:
            d[k] = 1
            return False
        return True

    # def func1(L):
    #     k3 = 0
    #     for i in range(3):
    #         for j in range(3):
    #             k3 *= 10
    #             k3 += L[i][j]
    #     if k3==123456789:
    #         print(0)
    #         exit()
    #     return k3
    if k3==123456789:
        print(0)
        exit()
    if str(k3)[:-3]=="987654":
        k3 = str(k3)[-3:]
        k3 = int(k3)
        bfs = [((k3,13))]
        for val,dist in bfs:
            for i in range(3):
                if i+1<3 and (i//3)==((i+1)//3):
                    k = str(val)
                    k = int(k[:i]+k[i+1]+k[i]+k[i+2:])
                    if not func(k):
                        if k==123:
                            print(dist+1)
                            exit()
                        bfs.append((k,dist+1))
        exit()

    bfs = [((k3,0))]
    for val,dist in bfs:
        for i in range(9):
            if i+1<9 and (i//3)==((i+1)//3):
                k = str(val)
                k = int(k[:i]+k[i+1]+k[i]+k[i+2:])
                if not func(k):
                    if k==123456789:
                        print(dist+1)
                        exit()
                    bfs.append((k,dist+1))
            if i+3<9:
                k = str(val)
                k= int(k[:i]+k[i+3]+k[i+1]+k[i+2]+k[i]+k[i+4:])
                if not func(k):
                    if k==123456789:
                        print(dist+1)
                        exit()
                    bfs.append((k,dist+1))

    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()