# Attention-is-all-you-need
This project is hard coded from scratch by me only using numpy library.
A numpy Implementation of the Transformer: Attention Is All You Need

Requirements                                                                                                                               
NumPy

Why This Project?

The preprocess is quoted from others' project. But as you can see, the deeplearning part of attention only use numpy!!!  If you are interested in write deeplearning piplines from scratch, and wanna to join me, please let me know, my email is fangshuming519@gmail.com

I took a mini batch for example, select = np.array([ 2732, 43567]).

After 300 round of training, the model can exactly do translation in this mini batch.

Training process:

loading vocabulary...
done
loading datasets...
done

-----------------------
Loss :  90.7660016041  in round  0

pred1: [2393 6223 7553 8341 6228 3346 6786 8073 8647  524]

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [2393 1905 2756 2024 8073 8647 8647 8647 8647  524]

truth2: [527 134   1   3   0   0   0   0   0   0]

-----------------------
-----------------------
Loss :  76.2423487219  in round  1

pred1: [527   0   0   0   0   0   0   0   0   0]

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [527   0   0   0   0   0   0   0   0   0]

truth2: [527 134   1   3   0   0   0   0   0   0]

-----------------------
-----------------------

Loss :  61.7235774638  in round  2

pred1: [527   0   0   0   0   0   0   0   0   0]

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [527   0   0   0   0   0   0   0   0   0]

truth2: [527 134   1   3   0   0   0   0   0   0]

-----------------------
-----------------------

Loss :  48.2334190853  in round  3

pred1: [527   0   0   0   0   0   0   0   0   0]

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [527   0   0   0   0   0   0   0   0   0]

truth2: [527 134   1   3   0   0   0   0   0   0]

-----------------------
-----------------------

Loss :  40.3689809274  in round  4

pred1: [527   0   0   0   0   0   0   0   0   0]

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [527   0   0   0   0   0   0   0   0   0]

truth2: [527 134   1   3   0   0   0   0   0   0]

-----------------------
-----------------------

Loss :  36.5467921043  in round  5

pred1: [527   0   0   0   0   0   0   0   0   0]

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [527   0   0   0   0   0   0   0   0   0]

truth2: [527 134   1   3   0   0   0   0   0   0]

-----------------------
-----------------------

Loss :  0.792950064915  in round  138

pred1: [ 527  447 6565    9   57  404    3    0    0    0] 

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [527 134   1   3   0   0   0   0   0   0]

truth2: [527 134   1   3   0   0   0   0   0   0]

-----------------------
-----------------------

Loss :  0.74010736516  in round  301

pred1: [  15  447 6565    9   57  404    3    0    0    0]

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [527 134   1   3   0   0   0   0   0   0]

truth2: [527 134   1   3   0   0   0   0   0   0]

-----------------------
-----------------------

Loss :  0.739964157087  in round  302

pred1: [  15  447 6565    9   57  404    3    0    0    0]

truth1: [  15  447 6565    9   57  404    3    0    0    0]

pred2: [527 134   1   3   0   0   0   0   0   0]

truth2: [527 134   1   3   0   0   0   0   0   0]
