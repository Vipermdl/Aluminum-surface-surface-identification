###TianChi   2018广东工业智造大数据创新大赛——智能算法赛（初赛代码）

#### 感谢大神Herbert95 原始baseline代码[分享](https://github.com/Herbert95/tianchi_lvcai)。

因复赛任务调整，随将代码放出，仅作个人备忘使用。

初赛最终排名：7/2972

Accuracy:    94.0%    and    59.42%

实验组成部分：

	在原始代码的基础上尝试样本不平衡采样，重新调整batchsize，调整输入图片大小，kv，focalloss等。

	最终解决方案：

		batchsize设为12，图片resize到550之后再随机裁成512
		测试时，将图片随机randomcrop 30次，然后投票产生最终的分类结果。


我在原始代码上做部分改进，代码很多没有写完整。如需原始结果复现，请联系我本人。

