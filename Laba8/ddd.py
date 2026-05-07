import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

n_samples, batch_size, num_steps = 1000, 100, 2000
X_data = np.random.uniform(1, 10, (n_samples, 1))

y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('linear-regression'):
    k = tf.Variable(tf.random_normal((1, 1)), name='slope')
    b = tf.Variable(tf.zeros((1,)), name='bias')
    y_pred = tf.matmul(x, k) + b

loss = tf.reduce_sum((y - y_pred) ** 2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(num_steps):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        
        _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b], 
                                             feed_dict={x: X_batch, y: y_batch})
        
        if (i + 1) % 100 == 0:
            print(f'Епоха {i+1}: Loss = {loss_val:.4f}, k = {k_val[0][0]:.4f}, b = {b_val[0]:.4f}')

    print(f"\nРезультат навчання: k = {k_val[0][0]:.4f}, b = {b_val[0]:.4f}")
