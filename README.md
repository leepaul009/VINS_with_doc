# VINS

PVQ中值离散形式：
$ p_{b_{i+1}}^w = p_{b_i}^w + v_{b_i}^w \delta t + \dfrac{1}{2} \bar{a_i} \delta t^2 $



$ \gamma_{i+1}^{b_k} 
= \gamma_{i}^{b_k} \otimes \gamma_{i+1}^{i} 
= \gamma_{i}^{b_k} \otimes \begin{bmatrix} 1 \cr \frac{1}{2} \bar{\omega_i} \delta t \end{bmatrix} $  

$\alpha_{i+1}^{b_k} 
= \alpha_i^{b_k} + \beta_i^{b_k} \delta t + \dfrac{1}{2} \bar{a_i} \delta t^2$

$ \beta_{i+1}^{b_k} 
= \beta_{i}^{b_k} + \bar{a_i} \delta t $


$\hat{a_i}$和$\hat{\omega_i}$是时刻i的传感器测量量, $\hat{a_{i+1}}$和$\hat{\omega_{i+1}}$是时刻i+1的传感器测量量：
$ \bar{a_i} = \frac{1}{2} [ q_i (\hat{a_i} - b_{a_i}) + q_{i+1} (\hat{a_{i+1}} - b_{\alpha_i}) ] $
$ \bar{\omega_i} = \frac{1}{2} (\hat{\omega_i} + \hat{\omega_{i+1}}) - b_{\omega_i} $












