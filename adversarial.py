import tensorflow as tf
from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# Loading in data
#####################
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=(x_train/255.)-.5
x_test=(x_test/255.)-.5

ty=OneHotEncoder().fit_transform(y_train)

# Loading in the model
#########################
sess=tf.Session()

mod1="alpha:0.0-beta:0.0-keepProb:0.5"
mod2="alpha0.08-beta0.0-keepProb0.5"
mod=mod1

loader=tf.train.import_meta_graph(mod+'.meta')
loader.restore(sess,mod)

# Loading in the variables
#############################
graph=tf.get_default_graph()
a4=graph.get_tensor_by_name('a4:0')
yhat=graph.get_tensor_by_name('yhat:0')
x=graph.get_tensor_by_name('x:0')
kp=graph.get_tensor_by_name('kp:0')

grady=[tf.gradients(a4[:,tmp],x,name='grad'+str(tmp))[0] for tmp in range(10)]

# Let's try it out
#####################

numImages=1000
attackclass=2

match=[]
accuracy=[]

print("Testing for model "+mod+", attackclass:"+str(attackclass))
for count in range(numImages):

    if count % 200==0:
        print(count)
    testim=count

    # Adversarial perturbation
    adv=np.sign(sess.run(grady[attackclass],feed_dict={x:x_test[testim:testim+1,:,:,:],kp:1}))

    orig=sess.run(yhat,feed_dict={x:x_test[testim:testim+1,:,:,:],kp:1}).argmax(axis=1)
    attacked=sess.run(yhat,feed_dict={x:x_test[testim:testim+1,:,:,:]+adv*.05,kp:1}).argmax(axis=1)
    correct=y_test[testim]

    match.append(attacked[0]==correct[0])
    accuracy.append(orig[0]==correct[0])


print("Original accuracy: "+str(np.sum(accuracy)/numImages))
print("Accuracy after attack: "+str(np.sum(match)/numImages))
