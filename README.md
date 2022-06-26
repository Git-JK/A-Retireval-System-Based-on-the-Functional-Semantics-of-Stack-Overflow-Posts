# A retrieval system based on Functional Semantics of Stack Overflow Posts
本仓库主要为本科论文《基于Stack Overflow问答帖功能语义的检索系统的设计与实现》的代码实现部分，主要想法为使用动词词组和词组模式提取问答帖标题所包含的显式功能语义，并使用基于Bert的Bertlets模型提取问答帖问题描述和回答所包含的隐式功能语义，并与要查询的问题的显式和隐式功能语义进行匹配，从而进行Stack Overflow问答帖推荐
src中为源代码，baseline中为AnswerBot方法的复现代码，system-front-end为系统前端代码压缩包
