

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class ECPRegressionLayer(Layer):
    """ECP Regression Layer (ECPRL)"""

    def __init__(self,n_xor, rank, n_outputs, **kwargs):
        super().__init__(**kwargs)
        self.n_xor = n_xor
        self.rank = rank  # CP-rank
        self.n_outputs = n_outputs  # output siz


    def build(self, input_shape):
        self.phi_length = input_shape[1]  # phi vector length

        #Xavier 初始化（均匀）
        scale_phi = tf.sqrt(6.0 / (self.phi_length + self.phi_length))  # fan_in=phi_length, fan_out=phi_length

        # shape: (rank, n_xor, phi_length)                         不可以初始化为其他顺序，例如(rank, phi_length, n_xor) ，即便后续通过transpose保证运算正确也不行！会训练不出来
        self.rank1_matrices = self.add_weight(
            name="rank1_matrices",
            shape=(self.rank, self.n_xor, self.phi_length),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-scale_phi,
                maxval=scale_phi,
                seed=0
            ),
            dtype=tf.float32
        )

        # 2. 初始化 output_tensors (400,n_outputs) - 400个独立(n_outputs,)向量
        scale_out = tf.sqrt(6.0 / (1 + self.n_outputs[0]))  # 假设n_outputs是单元素元组
        # shape: (rank, n_outputs)
        self.output_tensors = self.add_weight(
            name="output_tensors",
            shape=(self.rank, *self.n_outputs),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-scale_out,
                maxval=scale_out,
                seed=0
            ),
            dtype=tf.float32
        )

        # 偏置项
        self.bias = self.add_weight(
            name="bias",
            shape=(self.n_outputs),
            initializer="zeros",
            dtype=tf.float32
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cout = tf.zeros((batch_size, *self.n_outputs), dtype=tf.float32)

        chunk_size = min(int(400*30000*6/batch_size.numpy()/self.n_xor), self.rank)         #rank=400,batch_size=60000,n_xor=6是显存的瓶颈，因此可以以此来分批计算（若超过这个瓶颈，就将rank分多次计算）
        for start in range(0, self.rank, chunk_size):
            end = min(start + chunk_size, self.rank)
            cout = tf.reshape(tf.reduce_sum(tf.multiply(tf.reduce_prod(tf.einsum('bl,rnl->rbn', inputs, self.rank1_matrices[start:end]), axis=2), self.output_tensors[start:end]), axis=0),(batch_size, *self.n_outputs))
        return cout + self.bias



    def get_config(self):
        config = super().get_config()
        config.update({
            "rank": self.rank,
            "n_outputs": self.n_outputs
        })
        return config



class ETTRegressionLayer(Layer):
    """ETT Regression Layer (ETTRL)"""

    def __init__(self,n_xor,dropout, rank, n_outputs, **kwargs):
        super().__init__(**kwargs)
        self.n_xor = n_xor
        self.rank = rank  # TT-rank
        self.n_outputs = n_outputs  # output size
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.phi_length = input_shape[1]  # phi vector length

        self.core_is = []
        # shape: (rank_{i},rank_{i+1}, phi_length)
        for i in range(self.n_xor):
            core_i = self.add_weight(
                name="core_i",
                shape=(self.rank[i], self.rank[i+1], self.phi_length),
                initializer=tf.keras.initializers.glorot_uniform,
                dtype=tf.float32
            )
            self.core_is.append(core_i)


        # shape: (1, n_outputs)
        self.output_tensor = self.add_weight(
            name="output_tensors",
            shape=(1, *self.n_outputs),
            initializer=tf.keras.initializers.glorot_uniform,
            dtype=tf.float32
        )

        # 偏置项
        self.bias = self.add_weight(
            name="bias",
            shape=(self.n_outputs),
            initializer="zeros",
            dtype=tf.float32
        )

    def call(self, inputs,training=False):
        batch_size = tf.shape(inputs)[0]
        # cout = tf.zeros((batch_size, *self.n_outputs), dtype=tf.float32)

        flag = 0
        for core_i in self.core_is:
            if flag == 0:
                result = tf.einsum('ik,abk->iab',inputs,core_i)    #(batch_size,rank_0,rank_1)
                flag = 1
            else:
                temp = tf.einsum('ik,abk->iab',inputs,core_i)    #(batch_size,rank_{i},rank_{i+1})
                result = tf.matmul(result, temp)                             #(batch_size,rank_0,rank_{i+1})
        if training:
            result = self.dropout(result, training=training)
        result = tf.reshape(result,(batch_size, 1))                       #(batch_size, 1)
        cout = tf.multiply(result, self.output_tensor)                    #(batch_size, n_outputs)
        return cout + self.bias


    def get_config(self):
        config = super().get_config()
        config.update({
            "rank": self.rank,
            "n_outputs": self.n_outputs
        })
        return config






class ETTRegressionLayer_LSPUF(Layer):

    def __init__(self,n_xor,dropout, rank, n_outputs, **kwargs):
        super().__init__(**kwargs)
        self.n_xor = n_xor
        self.rank = rank  # TT-rank
        self.n_outputs = n_outputs  # output size
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.phi_length = input_shape[1] // self.n_xor # phi vector length

        self.core_is = []
        # shape: (rank_{i},rank_{i+1}, phi_length)
        for i in range(self.n_xor):
            core_i = self.add_weight(
                name="core_i",
                shape=(self.rank[i], self.rank[i+1], self.phi_length),
                initializer=tf.keras.initializers.glorot_uniform,
                dtype=tf.float32
            )
            self.core_is.append(core_i)


        # shape: (1, n_outputs)
        self.output_tensor = self.add_weight(
            name="output_tensors",
            shape=(1, *self.n_outputs),
            initializer=tf.keras.initializers.glorot_uniform,
            dtype=tf.float32
        )

        # 偏置项
        self.bias = self.add_weight(
            name="bias",
            shape=(self.n_outputs),
            initializer="zeros",
            dtype=tf.float32
        )

    def call(self, inputs,training=False):
        batch_size = tf.shape(inputs)[0]
        # cout = tf.zeros((batch_size, *self.n_outputs), dtype=tf.float32)

        flag = 0
        cnt = 0
        for core_i in self.core_is:
            if flag == 0:
                result = tf.einsum('ik,abk->iab',inputs[:,cnt*self.phi_length:(cnt+1)*self.phi_length],core_i)    #(batch_size,rank_0,rank_1)
                flag = 1
            else:
                temp = tf.einsum('ik,abk->iab',inputs[:,cnt*self.phi_length:(cnt+1)*self.phi_length],core_i)    #(batch_size,rank_{i},rank_{i+1})
                result = tf.matmul(result, temp)                             #(batch_size,rank_0,rank_{i+1})
            cnt += 1

        if training:
            result = self.dropout(result, training=training)
        result = tf.reshape(result,(batch_size, 1))                       #(batch_size, 1)
        cout = tf.multiply(result, self.output_tensor)                    #(batch_size, n_outputs)
        return cout + self.bias


    def get_config(self):
        config = super().get_config()
        config.update({
            "rank": self.rank,
            "n_outputs": self.n_outputs
        })
        return config











if __name__ == '__main__':
    # shape: (rank,n_xor, phi_length)
    rank1_matrices = tf.constant([[[2, 0, 1], [1, 1, 2]], [[1, 2, 0], [3, 1, 1]]], dtype=tf.float32)
    rank1_matrices = tf.transpose(rank1_matrices, [0, 2, 1])

    print("----------------------------------------------------------------------------")
    # shape: (rank, phi_length,n_xor)
    # rank1_matrices = tf.constant([[[2,1],[0,1],[1,2]],[[1,3],[2,1],[0,1]]],dtype=tf.float32)
    output_tensors = tf.constant([[2], [3]], dtype=tf.float32)
    print(rank1_matrices)
    print(output_tensors)
    inputs = tf.constant([[1, 1, 1]], dtype=tf.float32)  # (batch_size,phi_length)
    print(inputs)
    start = 0
    end = 2
    batch_size = tf.shape(inputs)[0]
    n_outputs = (1,)

    chunk_matrices = rank1_matrices[start:end]  # shape: (chunk, phi_length, n_xor)
    chunk_outputs = output_tensors[start:end]  # shape: (chunk, n_outputs)
    #
    prod_terms = tf.matmul(inputs, chunk_matrices)  # shape: (rank, batch_size, n_xor)
    print("点乘结果", prod_terms)
    prod_terms = tf.reduce_prod(prod_terms, axis=2)  # shape: (rank, batch_size)
    print("连乘结果", prod_terms)
    prod_terms = tf.multiply(prod_terms, chunk_outputs)  # shape: (rank, batch_size , n_outputs)
    print("乘上outputs", prod_terms)
    cout = tf.reshape(tf.reduce_sum(prod_terms, axis=0), (batch_size, *n_outputs))  # shape: (batch_size, n_outputs)
    cout = tf.reshape(tf.reduce_sum(
        tf.multiply(tf.reduce_prod(tf.matmul(inputs, rank1_matrices[start:end]), axis=2), output_tensors[start:end]),
        axis=0), (batch_size, *n_outputs))
    print(cout)

    print("----------------------------------------------------------------------------")
    #
    rank1_tnsrs = []
    rank1_tnsr = []
    rank1_tnsr.append(tf.constant([2, 0, 1], dtype=tf.float32))
    rank1_tnsr.append(tf.constant([1, 1, 2], dtype=tf.float32))
    rank1_tnsr.append(tf.constant([2], dtype=tf.float32))
    rank1_tnsrs.append(rank1_tnsr)

    rank1_tnsr = []
    rank1_tnsr.append(tf.constant([1, 2, 0], dtype=tf.float32))
    rank1_tnsr.append(tf.constant([3, 1, 1], dtype=tf.float32))
    rank1_tnsr.append(tf.constant([3], dtype=tf.float32))
    rank1_tnsrs.append(rank1_tnsr)

    x = tf.constant([1, 1, 1], dtype=tf.float32)
    x = tf.reshape(x, (-1, 3))

    cout = tf.zeros([1], tf.float32)

    for j in range(0, len(rank1_tnsrs)):
        tout = tf.multiply(tf.scalar_mul(1, tf.matmul(x, tf.reshape(rank1_tnsrs[j][0], [-1, 1]))),
                           tf.scalar_mul(1, tf.matmul(x, tf.reshape(rank1_tnsrs[j][1], [-1, 1]))))
        for k in range(2, len(rank1_tnsrs[j]) - 1):
            tout = tf.multiply(tout, tf.scalar_mul(1, tf.matmul(x, tf.reshape(rank1_tnsrs[j][k], [-1, 1]))))
        tout = tf.multiply(tout, tf.reshape(rank1_tnsrs[j][-1], [-1, 1]))
        cout = tf.add(cout, tout)
    print(cout)

