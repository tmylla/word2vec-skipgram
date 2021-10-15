

[简介](#简介)

[安装](#安装)

[目录结构](#目录结构)

[模型介绍](#模型介绍)

[运行方式](#运行方式)

[参考](#参考)



# 简介

使用PyTorch实现“Skip-gram”模型，提供“无负采样”、“有负采样”两种方式分别训练中/英文词向量。



# 安装

```sh
numpy==1.21.2
torch==1.8.2+cu102
```



# 目录结构

```python
Word2Vec
├── Data  # 数据集
│    ├── en.txt
│    ├── zh.txt
├── log  # 训练日志
├── model  # 保存模型
├── dataloader.py
├── model.py
├── trainer.py
├── utils.py
```



# 模型介绍

**CBOW 和 Skip-gram**：[Mikolov T, Chen K, Corrado G, et al. Efficient Estimation of Word Representations in Vector Space[J]. Computer Science, 2013.](https://arxiv.org/pdf/1301.3781.pdf)

- CBOW：根据上下文来预测当前词的概率，且上下文所有的词对当前词出现概率的影响的权重是一样的；
- Skip-gram：根据当前词来预测上下文概率，skip-gram模型有一个长度为 2c+1的滑动窗口。

一般来说，CBOW训练更快，但skip-gram学习的词向量更细致，当语料库中有大量低频词时，使用skip-gram学习比较合适。通常skip-gram准确率比CBOW高。

![image-20211015143432479](https://gitee.com/misite_J/blog-img/raw/master/img/image-20211015143432479.png)

Skip-gram网络结构：

- 输入层：中心词的one-hot编码；
- 隐藏层：包含M个结点（M为词嵌入维度），最终目标就是学习隐层的权重矩阵；
- 输出层：N维度的向量（N为词汇表的大小），每一维概率代表当前词是中心词上下文的概率大小。

![image-20211015152038119](https://gitee.com/misite_J/blog-img/raw/master/img/image-20211015152038119.png)

模型基于成对的单词进行训练，**训练样本是单词对 ( input word, output word )** ，其中input word和output word都是one-hot编码向量，最终模型的输出是一个概率分布。

模型训练完成后，矩阵 W 中第 i 行，W′ 中第 i 列，就可以作为词向量。这两个向量可以取其一，也可以相加或拼接。



**Hierarchical Softmax、Negative Sampling、Subsampling of Frequent Words**：[Mikolov T, Sutskever I, Chen K, et al. Distributed Representations of Words and Phrases and their Compositionality[J]. 2013, 26:3111-3119.](https://arxiv.org/pdf/1310.4546.pdf)

- Hierarchical Softmax：原始模型中 softmax 公式的分母部分需要遍历所有词，而词的数量常常是几十万的量级，这一步就会非常耗时间。

  - Hierarchical softmax 使用**二叉树**来记录所有的词，所有词被放在树的叶子节点（白色）上。每个内部节点（灰色）也有自己的词向量。基本想法是**每个叶子节点的概率，都是从根节点到该叶子节点的路径上记录的概率之积**。
  - 当输入为$w_I$ ，对于期望输出 $w_O$ 而言（概率较大），希望$w_I$的向量与根节点到$w_O$路径上所有节点的向量內积越大越好。
  - $p\left(w \mid w_{I}\right)=\prod_{j=1}^{L(w)-1} \sigma\left([ n(w, j+1)=\operatorname{ch}(n(w, j)) ] \cdot v_{n(w, j)}^{\prime}{ }^{\top} v_{w_{I}}\right)$，其中，$L(w)$ 为树的深度，方括号里面是一个示性函数：每次从根节点到$w_O$**路径上的所有节点中随机取一个，然后选一个孩子节点，**如果选则是路径上的节点，就返回 1，否则返回 -1。这么做就是希望处在路径上的节点的向量与$w_I$ 內积越大越好，不再路径上的节点內积越小越好。

  ![img](https://wangyu-name.oss-cn-hangzhou.aliyuncs.com/superbed/2019/09/01/5d6b8c29451253d178438ee1.jpg)

- Negative Sampling：也是为了缓解 softmax 计算量大的问题。

  - 在训练过程中，取窗口中的 context word 作为正例，在窗口外选取 K 个词作为反例，这样每次考虑到了词就很有限了，消除了 softmax 中求和的部分。
  - 对于窗口中心的词 $w_I$，窗口内周围的词 $w_O$ 以及随机抽取的词 $w_i$，最大化期望：$\log \sigma\left(v_{w_{O}}^{\prime}{ }^{\top} v_{w_{I}}\right)+\sum_{i=1}^{k} \mathbb{E}_{w_{i} \sim P_{n}(w)}\left[\log \sigma\left(-v_{w_{i}}^{\prime}{ }^{\top} v_{w_{I}}\right)\right]$。
    - $\mathbb{E}_{w_{i} \sim P_{n}(w)}$是指抽取的词 $w_i$ 服从某种分布，即$w_i$根据词频被抽到的概率为：$P_{\alpha}\left(w_{i}\right)=\frac{c\left(w_{i}\right)^{\alpha}}{\sum_{j}^{T} c\left(w_{j}\right)^{\alpha}}$；
    - 其中 $c(w)$ 是词 $w$ 出现的次数。$\alpha$是一个可调参数，用于对词频对高频词进行惩罚，对低频词进行补偿，论文中令$\alpha=0.75$.

- Subsampling of Frequent Words：在训练的时候，抛弃一些高频词，能够加快模型收敛速度，且让 Embedding 训练的更好。

  - 抛弃词 $w_i$的概率为$P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}$，其中，$f(w_i)$ 表示词 $w_i$ 出现的频率；$t$ 是人为选定的一个值，参考值是 $10^{−5}$，$t$ 选取的越小，抛弃的概率也就越小。



# 运行方式

`run.py`文件内设定以下参数后，运行该py文件即可。

```python
language = 'zh'
neg_sample = True  # 是否负采样
embed_dim = 300
C = 3  # 窗口大小
K = 15  # 负采样大小
num_epochs = 100
batch_size = 32
learning_rate = 0.025
```



# 参考

1. [Tutorial - Word2vec using pytorch](https://rguigoures.github.io/word2vec_pytorch/)
2. [PyTorch实现Word2Vec](https://cloud.tencent.com/developer/article/1613950)
3. [论文阅读 - Distributed Representations of Words](https://www.cnblogs.com/wy-ei/p/11534647.html)

