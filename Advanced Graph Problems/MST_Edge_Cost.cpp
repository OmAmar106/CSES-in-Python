// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
 
#define int long long
#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vector<int>>;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
 
template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<typename T>
class is_printable {
    template<typename U> static auto test(int) -> decltype(cout << declval<U>(), true_type{});
    template<typename> static auto test(...) -> false_type;
public: static constexpr bool value = decltype(test<T>(0))::value;
};
template<typename T> enable_if_t<is_printable<T>::value> __print(const T& val) { cout << val; }
template<typename T1, typename T2> void __print(const pair<T1, T2>& p) {
    cout << "("; __print(p.first); cout << ":"; __print(p.second); cout << ")";
}
template<typename T> enable_if_t<!is_printable<T>::value> __print(const T& container) {
    cout << "{"; bool first = true;
    for (const auto& x : container) { if (!first) cout << ", "; __print(x); first = false; }
    cout << "}";
}
void print() { cout << "\n"; }
template<typename T, typename... Args>
void print(const T& t, const Args&... rest) {
    __print(t); if constexpr (sizeof...(rest)) cout << " "; print(rest...);
}
 
int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}
 
// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, SortedList: sortl
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
 
class DisjointSetUnion {
public:
    vector<int> parent, size;
    vector<set<int>> flags;
    map<int,int> ans;
 
    DisjointSetUnion(int n) : parent(n), size(n, 1) {
        iota(parent.begin(), parent.end(), 0);
        flags.resize(n);
    }
 
    int find(int a) {
        if (parent[a] != a) {
            parent[a] = find(parent[a]);
        }
        return parent[a];
    }
 
    int unite(int a, int b, int w) {
        // print(a,b);
        a = find(a);
        b = find(b);
        if (a != b) {
            if (size[a] < size[b]) swap(a, b);
 
            if(len(flags[a])<len(flags[b])){
                swap(flags[a],flags[b]);
            }
            for(auto j:flags[b]){
                if(flags[a].find(j)!=flags[a].end()){
                    flags[a].erase(j);
                    ans[j] = w;
                }
                else{
                    flags[a].insert(j);
                }
            }
            flags[b].clear();
 
            parent[b] = a;
            size[a] += size[b];
            return w;
        }
        return 0;
    }
 
    int set_size(int a) {
        return size[find(a)];
    }
 
    int notfind(int a) {
        int k = find(a);
        for (int i = 0; i < parent.size(); i++) {
            if (find(i) != k) {
                return i;
            }
        }
        return -1;
    }
};
 
void solve() {
    int n,m;
    cin>>n>>m;
 
    DisjointSetUnion ds(n);
 
    vector<tuple<int,int,int,int>> L;
    range(i,m){
        int u,v,w;
        cin>>u>>v>>w;
        L.emplace_back(w,u-1,v-1,i);
        ds.flags[u-1].insert(i);
        ds.flags[v-1].insert(i);
    }
 
    sort(all(L));
    
    int ans1 = 0;
    for(auto [w,u,v,ind]:L){
        ans1 += ds.unite(u,v,w);
    }

    vector<int> ans(m);
    // print(ds.ans);
    for(auto [w,u,v,i]:L){
        ans[i] = ans1-ds.ans[i]+w;
            // print("YES");
    }
 
    for(auto it:ans){
        print(it);
    }
 
}
 
int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}