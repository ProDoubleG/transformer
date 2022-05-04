# transformer
Overall structure is from https://wikidocs.net/31379

modification was made on 
1. Attention calculation process using tf.einsum instead of tf.matmul
2. Added another parameter for dv, the deapth of value weight
![model](https://user-images.githubusercontent.com/78391621/166668013-c288ce2a-e799-4dd1-a62b-6fa3aa935f89.png)
