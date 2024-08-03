# Supervised Matrix Factorization #
## for simultaneous low-rank compressio and classification ##

<br/> This repository contains the scripts that generate the main figures reported in the paper: <br/>


Joowon Lee, Hanbaek Lyu, Weixin Yao
[*"Exponentially Convergent Algorithms for Supervised Matrix Factorization*"]([[https://arxiv.org/abs/2102.06984](https://papers.nips.cc/paper_files/paper/2023/hash/f2c80b3c9cf8102d38c4b21af25d9740-Abstract-Conference.html)] (NeurIPS 2023)

Joowon Lee, Hanbaek Lyu, Weixin Yao
[*"Supervised Matrix Factorization: Local Landscape Analysis and Applications*"]([[https://arxiv.org/abs/2102.06984](https://proceedings.mlr.press/v235/lee24p.html)] (ICML 2024)


&nbsp;
 

&nbsp;

![](Figures/Fig1.png)
&nbsp;
![](Figures/Fig2.png)
&nbsp;
![](Figures/Fig3.png)
&nbsp;
![](Figures/Fig4.png)
&nbsp;


## Usage

Please see the demo "notebook/SMF_gene_groups_demo.ipynb"
&nbsp;

Then copy & paste the ipynb notebook files into the main folder. Run each Jupyter notebook and see the instructions therein. 

## File description 

  1. **src.SMF.py** : Numpy implementation of SMF_BCD (Block Coordinate Descent, ICML '24) and SMF_LPGD (Low-rank PGD, NeurIPS '23)
  2. **src.SMF_torch.py**: Pytorch implementation of SMF_BCD (ICML '24). It utilizes GPU if available. 
  
## Authors

* **Joowon Lee** - *Initial work* - [Website](https://www.math.ucla.edu/~ykureh/)
* **Hanbaek Lyu** - *Initial work* - [Website](https://hanbaeklyu.com)
* **Weixin Yao** - *Initial work* - [Website](https://hanbaeklyu.com)

## Code Contributors 
* **Agam Goyal**
* $$Yee Wei**
* (Add yours if you make contributions!)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

