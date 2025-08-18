# Spherical Average (多点間の球面線形補間)について

こんにちは。GokRackこと極落にんじんです。

今回は、私が Beatrice VSTの[2.0.0-beta.3のアップデートとして話者のモーフィング機能 (Voice Morphing Mode)](https://x.com/prj_beatrice/status/1880871367975027188)を実装したときに用いた Spherical Average (多点間の球面線形補間) についてのお話です。

## 背景

私が Beatrice VST に興味を持ってソースコードを眺めたのは Ver.2.0.0-beta.2 の頃なのですが、
その当時のVSTのソースコードを読み解く限り、 Beatrice は話者の特徴量を 256 次元の Embeddings で保持しており、
話者切り替えは推論時に与える 256 次元のベクトルの値を複数切り替えることで実現されているということが読み取れました。

そこで、試しに VSTのソースコードに手を加えてそれら話者ごとの特徴量ベクトルの加重平均をとって推論させてみたところ、
上手いこと話者の特徴が混ざったような変換結果が得られました。

その結果に気を良くして VST の UI などの改造も加えて prj_beatrice さんにプルリクを送ったところ、
色々修正しつつも無事に採用されて 2.0.0-beta.3 としてリリースされたという流れになっています。

## Spherical Weighted Average

私が最初に prj_beatrice さんにプルリクを送った時点では、
話者モデルの加重平均手法としては単純な[線形補間 (Lerp)](https://ja.wikipedia.org/wiki/%E7%B7%9A%E5%BD%A2%E8%A3%9C%E9%96%93)を用いていました。
それに対して prj_beatrice さんが「[球面線形補間(Slerp)](https://en.wikipedia.org/wiki/Slerp)も検討したい」
というような反応を頂きました。

ただ、 Slerp は２つのベクトル同士の補間を行うことは出来ますが、３つ以上のベクトル同士の補間は出来ません。
そこで多点間で使えるように Slerp を拡張した手法は無いだろうかと探してみたところ、
[Spherical Averages](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/spheremean/index.html) 
という論文を見つけました。
この論文中の Spherical Weighted Average を実装したところ、
聴いてみた感じ変換結果も良くなった(ように感じられた)ため、最終的にこちらを採用しました。

https://mathweb.ucsd.edu/~sbuss/ResearchWeb/spheremean/index.html

Spherical Weighted Average は、単位超球面上のベクトル $p_i$を重み$w_i$をもとに単位超球面上で加重平均をとったベクトル $q$を、
単位超球面 S 上の２点間の距離を $\mathrm{dist}_S( p, q )$ を用いて
$\|\sum_{n=1}^N w_i \mathrm{dist}_S( p_i, q )^2\|$ を最小化する問題の解として
反復計算により算出するするという手法になっており、
それを計算する具体的なアルゴリズムとして A1 と A2 が論文中に収められています。

アルゴリズム A1 は１次収束のアルゴリズムで、ざっくり説明すると

1. 加重平均ベクトル $q$ の初期値 $q_0$ として、LERPによる計算結果のベクトルを単位超球面上まで伸ばしたもの $q_0 = \frac{ w^Tp }{\| w^T p \|}$ を用いる。
2. $t$回目の反復計算において、$q_t$から見た各 $p_i$ の方向と距離が等しくなるような、$q_t$ における超接平面上の点 $l_{q_t}(p)$ を求める。
3. $l_{q_t}(p)$ の加重平均 $w^T l_{q_t}(p)$ を計算し、その点に対応する元の単位超球面上の点を $q_{t+1} = \exp_{q_t}(q_t + u )$として値を更新する。
4. $w^T l_p(p) - q_t$ が十分小さくなったら反復計算を終了する。

といったものになっています。
なお、 $l_{q}(p)$は単位超球面上の点 p を同じく単位超球面上の点qにおける超接空間に射影する関数を表し、$\exp_{q}(p)$はその逆関数となっています。

アルゴリズム A2 は A1 の 3. の部分を

3. ヘッセ行列 H を用いて $ q_{t+1} = \exp_{q_t}( q_t + H^{-1}(w^T l_{q_t}(p)-q))$ として値を更新する。

というふうに変更することにより、収束速度を二次収束に早めたものになっています。

ただ、A2ではヘッセ行列の算出に回転行列が登場するのですが、
2次元や3次元の場合はともかく今回のような M 次元の空間での回転行列がちょっとわからなかったこと、
また逆行列演算が必要となるがその実装はちょっと面倒だし
１反復あたりの演算量が大きくなることからリアルタイム演算には向かないだろうと考え、
2.0.0-beta.3 時点でのBeatrice の話者マージでは A1 を実装して使用しました。

## Spherical Weighted Average の高速化について

2.0.0-beta.3 時点ではこのアルゴリズム A1 でも問題なさそうだったのですが、
将来的に話者特徴量が増えたらちょっと計算量的に厳しくなるような気がしてきました。

なので、もっと速く求めるアルゴリズムは無いかと考えてみました。

このアルゴリズム A1 を眺めてみると毎回の反復で $l_{q}(p)$ と $\exp_{q}(q+u)$を計算しているのですが、
ここが結構計算量的に重いです。

そこで大元の最適化問題に立ち返ってみます。
単位超球面上$S$の距離$\mathrm{dist}_S(p, q)$ は $\mathrm{dist}_S(p, q) = \arccos( p_i^T q )$ により計算できますので、
Spherical Weighted Average が解く最適化問題は
$$
\min_q \sum_i w_i \arccos( p_i ^T q )\\
\mathrm{s.t.} \|q\| = 1
$$
と表すことが出来ます。

この問題は q が単位超球面上にある (\|q\| = 1)という制約条件付きの最適化問題になっていますが、
このような最適化問題は [Riemann 多様体上の最適化問題](https://qiita.com/wsuzume/items/3f391369330abefbdb41)として知られ、
通常の制約条件なしの最適化問題を解く
[最急降下法](https://ja.wikipedia.org/wiki/%E6%9C%80%E6%80%A5%E9%99%8D%E4%B8%8B%E6%B3%95)や
[L-BFGS法](https://ja.wikipedia.org/wiki/L-BFGS%E6%B3%95)
に少し手を加えるだけで解けることが知られています。

https://qiita.com/wsuzume/items/3f391369330abefbdb41

この目的関数 $ L(q) = \sum_i w_i \arccos( p_i ^T q )$ の $q$ についての勾配を考えてみると
$$
\nabla_q L(q) = - 2 \sum_i \frac{  w_i p_i \arccos( p_i ^T q ) }{\sqrt{1-(p_i^T q)^2}}
$$
と意外と簡単に計算できそうな形をしています。

そこで Riemann 多様体上の最適化問題用に修正した
[最急降下法](https://ja.wikipedia.org/wiki/%E6%9C%80%E6%80%A5%E9%99%8D%E4%B8%8B%E6%B3%95)や
[L-BFGS法](https://ja.wikipedia.org/wiki/L-BFGS%E6%B3%95)を python で実装し、
計算時間と計算結果を比較したところ、下記のような結果が得られました。

- 計算条件
  - ベクトル数 $N = 256$
  - 次元数 $M = 256 $
  - 重み $w_i$ : 0~1 の一様乱数によって計算したあと、 $\sum_i w_i = 1 $となるように正規化
  - ベクトル $p_i$ : 平均 0 標準偏差 1 の正規分布乱数によって計算したあと、 $\|p_i\| = 1$　となるように正規化
  - 異なる重みとベクトルについて 1000 回計算し、その平均所要時間を算出
  - 計算には 32bit float を使用(VST内部が32bit floatなので)
    - そのため、$10^{-7}$ 程度の演算誤差が見込まれる。

- 計算環境
  - OS : Windows 11Pro 24h2
  - CPU : Ryzen 5600X
  - GPU : RTX 2080 Ti
  - Mem : DDR4-2933 32GB
  - Python 3.12.6, numpy 2.2.0

- 結果
  - アルゴリズム A1
    - 平均反復回数 : 89.16 回
    - 平均所要時間 : 7.72 ms
    - １反復あたりの平均所要時間 : 0.0866 ms
  - 最急降下法
    - 平均反復回数 : 44.94 回
    - 平均所要時間 : 1.89 ms
    - １反復あたりの平均所要時間 : 0.0421 ms
    - A1 の結果と比較しての q と v の最大誤差 : 1.788139e-07
  - L-BFGS法
    - 平均反復回数 : 5.06 回
    - 平均所要時間 : 0.43 ms
    - １反復あたりの平均所要時間 : 0.0859 ms
    - A1 の結果と比較しての q と v の最大誤差 : 3.129244e-07

なお、実装した python スクリプトは下記場所に置いております。

https://github.com/Mark-GokRack/spherical_averages/blob/main/python/sph_avg.py

最急降下法、L-BFGS法ともに、元アルゴリズム A1 に比べて圧倒的に高速化出来てます。
得られた値の誤差についても 32bit float と同程度に収まっているため、得られた値にも問題はなさそうですね。

A1 に比べて最急降下法で、超接平面への射影などが減ることから１反復あたりの所要時間が減ることは予想していたのですが、
反復回数それ自体が半分くらいに減るのは嬉しい誤算でした。
最急降下法からL-BFGS法にするときの１反復あたりの所要時間の増加するだろうとは思っていましたが、
増えても A1 と同程度であることからそのまま A1 を L-BFGS法で置き換えてしまっても問題なさそうに感じます。

## Appendix.

### Spherical Weighted Average の話者特徴量のモーフィングへの適用方法

Spherical Weighted Average は複数の単位ベクトル同士の間で球面上の加重平均を求める手法である一方、
Beatrice の話者特徴量は必ずしも単位ベクトルではありません。

そこで、 

1. Beatrice の話者特徴量ベクトル $p_i^{\prime}$ と同じ方向の単位ベクトル $p_i = p_i^{\prime} / \|p_i^{\prime}\|$
と話者ごとの重み係数 $w_i$ ($\sum_i w_i = 1$を満たす)を用いて Spherical Weighted Average によって加重平均ベクトル $q$ を求める
2. $\sum_i v_i p_i = q$ となるような重み係数 $v_i$ を求める
3. $\sum_i v_i p_i^{\prime}$ を計算してモーフィング後の話者特徴として使用する

と言った流れで話者特徴量ベクトルをモーフィングしています。

^[Slerp を用いたモデルマージでも、ベクトル $p'$ を単位ベクトル化したベクトル$p$を用いて求めた重み係数$v$をもとのに適用して加重平均をとっていることに相当するため、このような流れで大丈夫だろうと判断しました。]

この $\sum_i v_i p_i = q$ となるような重み係数 $v_i$を求め方について考えます。
まず、先述の単位超球面の点 p を点q における超接空間に射影する関数 $l_q(p)$を具体的に書き下すと、 $p$ と $q$ のなす角 $\theta = \arccos( p^T q)$ として下記のようになります。
$$
l_q(p) = \frac{\theta}{\sin \theta} p + \left( 1 - \frac{\theta \cos \theta}{\sin\theta}\right) q
$$

また、論文よりうまく収束した場合は $\sum_i w_i l_q(p_i) = q$ となることが示されています。

このことから  $\sum_i w_i l_q(p_i) - q$ について考えると、
$$
\sum_i w_i l_q(p_i) - q = \sum_i\frac{w_i\theta_i}{\sin \theta_i} p_i  - \sum_i\frac{w_i \theta_i \cos \theta_i}{\sin\theta_i} q = 0
$$
両辺を $\sum_n\frac{w_n \theta_n \cos \theta_n}{\sin\theta_n}$ で割って
^[厳密にはここで $\sum_n\frac{w_n \theta_n \cos \theta_n}{\sin\theta_n} \neq 0$ であることを確認する必要がある気がしますが、加重平均を取るベクトルの数 N がベクトルの次元数 M 以下の場合は多分大丈夫じゃないかと思います]
$$
\sum_i\frac{w_i\theta_i}{\sin \theta_i \sum_n\frac{w_n \theta_n \cos \theta_n}{\sin\theta_n}} p_i  - q = 0
$$
$$
\therefore v_i = \frac{w_i\theta_i}{\sin \theta_i \sum_n\frac{w_n \theta_n \cos \theta_n}{\sin\theta_n}}
$$

として求められます。

^[ただし、この値を用いることができるのは平均を取る点の数 N がベクトルの次元数 M 以下の場合に限られます。
M > N の場合は本質的に劣決定問題となり、$\sum_i v_i p_i = q$ を満たす $v_i$ をが一意に定まらず無数に存在し、
$\sum_n\frac{w_n \theta_n \cos \theta_n}{\sin\theta_n}$ の値が $0$ に近くなったりするため 
上記の式で計算される $v_i$ が異常に大きな値になったりしてしまいます。
その場合は、$p_i$を並べた行列 $P$ を用いて、$Pv = q$の最小ノルム解$v = P^T(PP^T)^{-1} q $を採用する必要があります。]


### ２点間の場合の Spherical Average と Slerp の一致確認

念の為、Spherical Weighted Average がちゃんと Slerp の多点間拡張に相当するのかどうかを確認するために、
２点間の Spherical Weighted Average と Slerp の比較を考えます。

２点の場合、重み係数は $w_1 = 1-t$、$w_2 = t$ と表すことが出来ます。
また、Slerp の場合、$p_1$ と $p_2$ の成す角を $\Omega$ として $\theta_1 = t\Omega$, $\theta_2 = (1-t)\Omega$ を満たす点 $q$ が点 $p_i$ による加重平均値となります。

この$w_1, w_2, \theta_1, \theta_2$ の関係性を上記の $v_i$ の式に適用すると、色々代入したり通分したりしていった結果
$$
v_1 = \frac{\sin\theta_2}{\cos\theta_1\sin\theta_2 + \cos\theta_2\sin\theta_1} = \frac{\sin\left(\left(1-t\right)\Omega\right)}{\sin\Omega}
$$
$$
v_2 = \frac{\sin\theta_1}{\cos\theta_1\sin\theta_2 + \cos\theta_2\sin\theta_1} = \frac{\sin(t\Omega)}{\sin\Omega}
$$
となるのですが、これは [Slerpの定義式](https://en.wikipedia.org/wiki/Slerp)に一致するため、$\sum_i v_i p_i = q$ が満たされます。

このとき、$\sum_i w_i l_q(p_i) - q$ について考えると、
$$
\sum_i w_i l_q(p_i) - q = \sum_i\frac{w_i\theta_i}{\sin \theta_i} p_i - q \sum_i\frac{w_i\theta_i \cos \theta_i}{\sin\theta_i} \\
= \left(\sum_i\frac{w_i\theta_i \cos \theta_i}{\sin\theta_i}\right) \left( \sum_i v_i p_i - q\right)  = 0
$$
となることから $\sum_i w_i l_q(p_i) = q$ も満たされるため、 Slerp による加重平均値は Spherical Weighted Average の収束条件を満たすことがわかります。

よって、２点間の Spherical Weighted Average は Slerp の解に収束するため、両者は同じものであると考えて良さそうです。

