// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <cstddef>
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
// Segment Tree(Nodes): SegmentNode, HLD: hld, fwht: fwht
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

struct Node {
    int val, pr, sz,rev;
    Node *l, *r,*par;
    Node(int v) : val(v), pr(rng()), sz(1), l(nullptr), r(nullptr),rev(0),par(nullptr) {}
};

int sz(Node* t) { return t ? t->sz : 0; }

void upd(Node* t) {
    if (!t) return;
    t->sz = 1 + sz(t->l) + sz(t->r);
    if (t->l) t->l->par = t;
    if (t->r) t->r->par = t;
}

void push(Node* t) {
    if (!t || !t->rev) return;
    swap(t->l, t->r);
    if (t->l) t->l->rev ^= 1;
    if (t->r) t->r->rev ^= 1;
    t->rev = 0;
}

void push_to_root(Node* t) {
    vector<Node*> stk;
    while (t) {
        stk.push_back(t);
        t = t->par;
    }
    while (!stk.empty()) {
        push(stk.back());
        stk.pop_back();
    }
}

Node* merge(Node* a, Node* b) {
    if (!a || !b) return a ? a : b;

    push(a);
    push(b);
    if (a->pr > b->pr) {
        a->r = merge(a->r, b);
        upd(a);
        return a;
    } else {
        b->l = merge(a, b->l);
        upd(b);
        return b;
    }
}

pair<Node*,Node*> split(Node* t, int k) {
    if (!t) return {nullptr, nullptr};
    push(t);
    if (sz(t->l) >= k) {
        auto p = split(t->l, k);
        t->l = p.second;
        upd(t);
        if (p.first) p.first->par = nullptr;
        return {p.first, t};
    } else {
        auto p = split(t->r, k - sz(t->l) - 1);
        t->r = p.first;
        upd(t);
        if (p.second) p.second->par = nullptr;
        return {t, p.second};
    }
}

int index(Node* t) {
    push_to_root(t);
    int ans = sz(t->l);
    while (t->par) {
        Node* p = t->par;
        if (t == p->r) {
            ans += 1 + sz(p->l);
        }
        t = p;
    }
    return ans;
}

void inorder(Node* t, string &s, string &out) {
    if (!t) return;
    inorder(t->l, s, out);
    out += s[t->val];
    inorder(t->r, s, out);
}

void solve() {
    int n;
    cin>>n;
    vi L = LII(n);

    Node* st = NULL;
    map<int,Node*> d;

    for(auto it:L){
        Node* k = new Node(it);
        d[it] = k;
        st = merge(st,k);
    }
    vector<pair<int,int>> ans;
    for(int i=0;i<n;i++){
        int f = index(d[i+1]);
        if(f!=i){
            ans.emplace_back(i+1,f+1);
            auto [node1,node2] = split(st,i);
            auto [node3,node4] = split(node2,f-i+1);
            node3->rev ^= 1;
            node2 = merge(node3,node4);
            st = merge(node1,node2);
        }
    }

    print(len(ans));
    for(auto [a,b]:ans){
        print(a,b);
    }

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}