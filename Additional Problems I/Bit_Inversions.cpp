// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <queue>
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
// #pragma GCC optimize("O3,unroll-loops")
// #pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
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

struct Node {
    int l0, r0, l1, r1, maxi;
    bool is1, is0;

    Node() {}

    Node(int l0, int r0, int l1, int r1, int maxi, bool is1, bool is0)
        : l0(l0), r0(r0), l1(l1), r1(r1), maxi(maxi), is1(is1), is0(is0) {}

    static Node merge(const Node* left, const Node* right) {
        if (!right) return *left;
        if (!left) return *right;

        const Node &L = *left, &R = *right;

        return Node(
            max(L.l0, L.is0 ? L.l0 + R.l0 : 0),
            max(R.r0, R.is0 ? R.r0 + L.r0 : 0),
            max(L.l1, L.is1 ? L.l1 + R.l1 : 0),
            max(R.r1, R.is1 ? R.r1 + L.r1 : 0),
            max({L.maxi, R.maxi, L.r0 + R.l0, L.r1 + R.l1}),
            L.is1 & R.is1,
            L.is0 & R.is0
        );
    }

    void update() {
        is0 = !is0;
        is1 = !is1;
        l0 = 1 - l0;
        r0 = 1 - r0;
        l1 = 1 - l1;
        r1 = 1 - r1;
    }
};

class SegmentTree {
public:
    int n;
    vector<Node> tree;
    vector<int> lazy_add;
    vector<int> lazy_set;
    vector<bool> has_set;

    SegmentTree(vector<int>& data) {
        n = 1;
        while(n<(int)data.size()){
            n <<= 1;
        }
        tree.resize(2*n,Node(0,0,0,0,0,1,1));
        // lazy_add.assign(4*n, 0);
        // lazy_set.assign(4*n, 0);
        // has_set.assign(4*n, false);
        for (int i = 0; i < data.size(); i++) {
            tree[n+i] = Node(1-data[i],1-data[i],data[i],data[i],1,data[i],1-data[i]);
        }
        for (int i = n-1; i >= 1; i--) {
            tree[i] = Node::merge(&tree[2*i], &tree[2*i+1]);
        }
    }

    // void apply_set(int idx, int l, int r, int val) {
    //     tree[idx].set(val * (r - l + 1));
    //     lazy_set[idx] = val;
    //     has_set[idx] = true;
    //     lazy_add[idx] = 0;
    // }

    // void apply_add(int idx, int l, int r, int val) {
    //     tree[idx].add(val * (r - l + 1));
    //     if (has_set[idx]) lazy_set[idx] += val;
    //     else lazy_add[idx] += val;
    // }

    // void push(int idx, int l, int r) {
    //     if (l == r) return;
    //     int mid = (l + r) / 2;

    //     if (has_set[idx]) {
    //         apply_set(idx*2, l, mid, lazy_set[idx]);
    //         apply_set(idx*2+1, mid+1, r, lazy_set[idx]);
    //         has_set[idx] = false;
    //     }
    //     if (lazy_add[idx]) {
    //         apply_add(idx*2, l, mid, lazy_add[idx]);
    //         apply_add(idx*2+1, mid+1, r, lazy_add[idx]);
    //         lazy_add[idx] = 0;
    //     }
    // }

    // Range Update in [L,R] if flag, then add
    void update(int idx) {
        idx += n;
        tree[idx].update();
        while(idx>1){
            idx >>= 1;
            tree[idx] = Node::merge(&tree[2*idx],&tree[2*idx+1]);
        }
    }
};

void solve() {
    string st;
    cin>>st;
    int n;
    cin>>n;
    vi L(n);
    cin>>L;
    vi L1;
    for(auto it:st){
        L1.pb(it-'0');
    }
    SegmentTree seg = SegmentTree(L1);
    
    for(auto it:L){
        seg.update(it-1);
        cout<<(seg.tree[1].maxi)<<" ";
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}