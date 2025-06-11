// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
using namespace std;
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define printf(x) cout << x << endl;

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

int parent[200000][30] = {0};

void solve() {
    int n,q;
    cin>>n>>q;

    for(int i=0;i<n;i++){
        int x;cin>>x;
        parent[i][0] = --x;
    }

    for(int i=1;i<30;i++){
        for(int j=0;j<n;j++){
            parent[j][i] = parent[parent[j][i-1]][i-1];
        }
    }

    while(q--){
        int a,k;
        cin>>a>>k;
        a--;
        int i = 0;
        while(k){
            if(k&(1)){
                a = parent[a][i];
            }
            i++;
            k >>= 1;
        }
        printf(++a)
    }
}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}