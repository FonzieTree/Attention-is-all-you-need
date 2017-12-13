# Attention-is-all-you-need
This project is hard coded from scratch by me only using numpy library.
A numpy Implementation of the Transformer: Attention Is All You Need

Requirements                                                                                                                               
NumPy

Why This Project?

The preprocess is quoted from others' project. But as you can see, the deeplearning part of attention only use numpy!!!  If you are interested in write deeplearning piplines from scratch, and wanna to join me, please let me know, my email is fangshuming519@gmail.com

Training process:
Loss :  38.96563723  in round  548                                                                                                         
pred1
[15  3  3  3  3  3  1  3  0  0] 
y1
[1173  732   35    5  718    8 3735    3    0    0]

pred2
[15  3  3  3  3  3  3  3  3  0]
y2
[120 807   5  29 143  91   5  77   3   0]

pre3
[15  3  8  1  3  0  0  0  0  0]
y3
[11 21  8  1  3  0  0  0  0  0]

pred4
[15  3  3  3  1  3  0  0  0  0]
y4
[1548 1199 1028    8 2495    3    0    0    0    0]

Loss :  29.8520902803  in round  957                                                                                                     
pred: [15 12  1  0  3  0  0  0  0  0]    truth: [  80 1252  987  891    3    0    0    0    0    0]                                     
pred: [15 13  3  0  0  0  0  0  0  0]    truth: [203  13   3   0   0   0   0   0   0   0]                                               
pred: [15 11 12  8  1  3  3  3  3  0]    truth: [ 15  17  12  28  73 219  59  23   3   0]                                               
pred: [15  1  1  3  4  1  3  0  0  0]    truth: [ 120   71 2595   10    4  609    3    0    0    0]                                       
