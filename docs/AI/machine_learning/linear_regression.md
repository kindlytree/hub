# Linear Regression
- [stanford ufldl Linear Regression](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/)
- [stanford cs229 ](http://cs229.stanford.edu/)
- [stanford cs229 notes1](http://cs229.stanford.edu/notes2019fall/cs229-notes1.pdf)
- [sample code1](https://github.com/imdeepmind/LinearRegressionInPytorch/blob/master/Linear_Regression_in_Pytorch.ipynb)
- [sample code2](https://github.com/kindlytree/ai/blob/master/samples/pytorch/basics/autograd.ipynb)

## Principle
Our goal in linear regression is to predict a target value y starting from a vector of input values [latex]x \in \Re ^n [/latex], For example, we might want to make predictions about the price of a house so that y represents the price of the house in dollars and the elements [latex]x_j[/latex] of x represent “features” that describe the house (such as its size and the number of bedrooms). Suppose that we are given many examples of houses where the features for the i’th house are denoted [latex]x^{(i)}[/latex] and the price is [latex]y^{(i)}[/latex]. For short, we will denote the
Our goal is to find a function [latex]y=h(x)[/latex] so that we have [latex]y^{(i)}≈h(x^{(i)})[/latex] for each training example. If we succeed in finding a function h(x) like this, and we have seen enough examples of houses and their prices, we hope that the function h(x) will also be a good predictor of the house price even when we are given the features for a new house where the price is not known.

To find a function h(x) where y(i)≈h(x(i)) we must first decide how to represent the function h(x). To start out we will use linear functions: [latex]h_θ(x)=\sum_j\theta_j x_j=θ⊤x[/latex]. Here, [latex]h_\theta(x)[/latex] represents a large family of functions parametrized by the choice of θ. (We call this space of functions a “hypothesis class”.) With this representation for h, our task is to find a choice of θ so that[latex] h_\theta(x^{(i)})[/latex] is as close as possible to [latex]y(i)[/latex]. In particular, we will search for a choice of θ that minimizes:

[latex]J(\theta)=\frac{1}{2}\sum_i(h_\theta(x^{(i)})−y(i))^2=\frac{1}{2}\sum_i(\theta^Tx^{(i)}−y^{(i)})^2[/latex]
This function is the “cost function” for our problem which measures how much error is incurred in predicting y(i) for a particular choice of θ. This may also be called a “loss”, “penalty” or “objective” function.

## Probability interpretation
当面临线性回归的时候，为什么将最小二乘作为成本函数是合适的呢？在本部分，将给出一个概率解释，就理解为什么最小二乘作为线性回归的成本函数是很自然的一个算法。  
  假设我们的目标变量和输入变量通过如下的等式关联起来:
[latex]
y^{(i)}=\theta ^Tx^{(i)}+\epsilon ^{(i)}
[/latex]
  其中[latex]\epsilon ^{(i)}[/latex]为误差项，可以将误差归结为是由于没有纳入到建模中的一些变量（特征），或者仅仅是由于噪声引起的。我们进一步假设[latex]\epsilon ^{(i)}[/latex]服从IID(independently and identically distributed)的均值为0，方差为[latex]\sigma[/latex]的高斯分布，写作 [latex]\epsilon ^{(i)}\sim N(0,\sigma^2) [/latex]。[latex]\epsilon ^{(i)}[/latex]的概率密度为：
[latex]p(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(\epsilon^{(i)})^2}{2\sigma^2})[/latex]
  因此下面的等式也会成立：
[latex]p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})[/latex]给定一个数据集，它的似然函数(样本的联合分布)可以表示为：  
[latex] L(\theta)=L(\theta;X,\vec y)=p(\vec y |X;\theta)[/latex]  
   由于我们假设[latex]\epsilon ^{(i)}[/latex]服从IID的高斯分布，因此上述的公式也可以写为:
[latex]L(\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta)=\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})[/latex]
  现在我们考虑怎么选择合适的[latex]\theta[/latex]来最大化[latex]L(\theta)[/latex]。这里面我们采用了一个等价的方法，通过最大化[latex]logL(\theta)[/latex]
[latex]l(\theta)=logL(\theta)= log\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})=\sum_{i=1}^{m}log\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})= mlog\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\frac{1}{2}\sum_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2[/latex]  
   根据上式，最大化[latex]l(\theta)[/latex]也就是最小化[latex]\frac{1}{2}\sum_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2[/latex]