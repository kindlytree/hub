# 深入理解GBDT多分类算法
- [原文链接](http://101.132.45.94/2020/01/29/understanding-gbdt-multi-class-classification-algorithm/)
- [参考文献](https://mp.weixin.qq.com/s/t2B5dg8uELNqSJcENHiwNA)
- [stanford tutorial SoftmaxRegression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
- [深入深出Sigmoid与Softmax的血缘关系](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247484122&idx=1&sn=41628bf3169b9ef3fa107646d483bae5&chksm=970c2a0ca07ba31ae1939e316c15695c83556c347e0b38bb80dde3048533de7de388ec2a6544&scene=21#wechat_redirect)
## GBDT 多分类算法

###  Softmax回归的对数损失函数
  当使用逻辑回归处理多标签的分类问题时，如果一个样本只对应于一个标签，我们可以假设每个样本属于不同标签的概率服从于几何分布，使用多项逻辑回归（Softmax Regression）来进行分类：
  [latex]P(Y=y_{i} | x) = h_{\theta}(x) = \begin{bmatrix}
P(Y=1|x;\theta)\\ 
P(Y=2|x;\theta)\\ 
......\\ 
P(Y=k|x;\theta)
\end{bmatrix} = \frac{1}{\sum_{j=1}^{k}e^{\theta_{j}^Tx}}\begin{bmatrix}
e^{\theta_{1}^Tx}\\ 
e^{\theta_{2}^Tx}\\ 
...\\ 
e^{\theta_{3}^Tx}
\end{bmatrix}
  [/latex]
  其中，[latex]\theta_1,\theta_2,...,\theta_k \in \Re^n [/latex]为模型的参数，而[latex]\frac{1}{\sum_{j=1}^{k}e^{\theta_{j}^Tx}}[/latex]可以看做是对概率的归一化。一般来说，多项逻辑回归具有参数冗余的特点，