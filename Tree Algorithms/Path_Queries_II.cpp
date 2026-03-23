// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;

#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define int long long
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vi>;
using viii = vector<vii>;
using ll = long long;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#ifdef LOCAL
#define dbg(x...) cout<<"["<<#x<<"] = "; print(x);
#else
#define dbg(x...)
#endif

template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<class T> bool chmin(T& a,const T& b){ if(b<a){ a=b; return 1;} return 0; }
template<class T> bool chmax(T& a,const T& b){ if(a<b){ a=b; return 1;} return 0;}
template<class T> using minpq = priority_queue<T,vector<T>,greater<T>>;
template<class T> using maxpq = priority_queue<T>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<class T> void _print(const T &x){ cout << x; }
template<class T,class U> void _print(const pair<T,U> &p){cout<<"(";_print(p.first);cout<<",";_print(p.second);cout<<")";}
template<class T>
void _print(const vector<T> &v){
    cout<<"[";bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"]";
}
template<class T>
void _print(const set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T>
void _print(const unordered_set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const unordered_map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
void print(){ cout<<"\n"; }
template<class T,class... Args>
void print(const T& a,const Args&... args){
    _print(a);
    if constexpr(sizeof...(args)) cout<<" ";
    print(args...);
}

int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}
int rnd(int l,int r){return uniform_int_distribution<int>(l,r)(rng);}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable, SortedList: sortl
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree, Trie/Treap: Tries
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, Centroid Decomp.: graph_decom
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, GrapthOth: graphoth, DSU: DSU, FFT:fft
// Rollback/Par DSU: rbdsu, treap: treap, graphflow(mat_match): graphflow, Persistent Seg Tree: perseg
// Segment Tree(Nodes): SegmentNode
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

struct HLD {
    int n, N;
    vector<vector<int>> adj;
    vector<int> parent, depth, size, heavy, head, pos, seg;

    HLD(vector<vector<int>>& g, vector<int>& val) {
        adj = g;
        n = adj.size();

        parent.assign(n, -1);
        depth.assign(n, 0);
        size.assign(n, 0);
        heavy.assign(n, -1);
        head.assign(n, 0);
        pos.assign(n, 0);

        stack<tuple<int,int,int>> st;
        st.push({0, -1, 0});

        while (!st.empty()) {
            auto [u, p, state] = st.top(); st.pop();
            if (state == 0) {
                st.push({u, p, 1});
                for (int v : adj[u]) {
                    if (v == p) continue;
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    st.push({v, u, 0});
                }
            } else {
                size[u] = 1;
                int mx = 0;
                for (int v : adj[u]) {
                    if (v == p) continue;
                    size[u] += size[v];
                    if (size[v] > mx) {
                        mx = size[v];
                        heavy[u] = v;
                    }
                }
            }
        }

        int cur = 0;
        stack<pair<int,int>> st2;
        st2.push({0, 0});

        while (!st2.empty()) {
            auto [u, h] = st2.top(); st2.pop();
            while (u != -1) {
                head[u] = h;
                pos[u] = cur++;

                int hu = heavy[u];
                for (int v : adj[u]) {
                    if (v != parent[u] && v != hu)
                        st2.push({v, v});
                }
                u = hu;
            }
        }

        N = 1;
        while (N < n) N <<= 1;
        seg.assign(2*N, 0);

        for (int i = 0; i < n; i++)
            seg[N + pos[i]] = val[i];

        for (int i = N - 1; i > 0; i--)
            seg[i] = max(seg[2*i], seg[2*i+1]);
    }

    void update(int u, int x) {
        int i = pos[u] + N;
        seg[i] = x;
        for (i >>= 1; i; i >>= 1)
            seg[i] = max(seg[2*i], seg[2*i+1]);
    }

    int query(int u, int v) {
        int res = 0;

        while (head[u] != head[v]) {
            if (depth[head[u]] < depth[head[v]])
                swap(u, v);

            int l = pos[head[u]] + N;
            int r = pos[u] + N + 1;

            while (l < r) {
                if (l & 1) res = max(res, seg[l++]);
                if (r & 1) res = max(res, seg[--r]);
                l >>= 1; r >>= 1;
            }

            u = parent[head[u]];
        }

        if (depth[u] > depth[v]) swap(u, v);

        int l = pos[u] + N;
        int r = pos[v] + N + 1;

        while (l < r) {
            if (l & 1) res = max(res, seg[l++]);
            if (r & 1) res = max(res, seg[--r]);
            l >>= 1; r >>= 1;
        }

        return res;
    }
};

void solve() {
    int n,q;
    cin>>n>>q;
    vi L = LII(n);
    vii d(n);
    for(int i=0;i<n-1;i++){
        int u,v;
        cin>>u>>v;
        u--;v--;
        d[u].pb(v);
        d[v].pb(u);
    }

    HLD hld = HLD(d,L);
    vi ans;

    while(q--){
        int ty;
        cin>>ty;
        if(ty==1){
            int u,x;
            cin>>u>>x;
            u--;
            hld.update(u,x);
        }
        else{
            int u,v;
            cin>>u>>v;
            u--;v--;
            ans.pb(hld.query(u,v));
        }
    }

    for(auto it:ans){
        cout<<it<<" ";
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}