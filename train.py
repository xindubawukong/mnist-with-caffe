


import caffe
from matplotlib import pyplot as plt



solver=caffe.get_solver('lenet_solver.prototxt')

# print([(k, v.data.shape) for k, v in solver.net.blobs.items()])
# print(type(solver.net.blobs))
# print([(k, v[0].data.shape) for k, v in solver.net.params.items()])
# solver.test_nets[0].forward()
# print(solver.net.blobs['data'].data.dtype)
# print(solver.net.blobs['data'].data[0,0])
# print(solver.net.blobs['label'].data[:100])
# plt.imshow(solver.net.blobs['data'].data[:8,0].transpose(1,0,2).reshape(28,8*28),cmap="gray");plt.axis('off');plt.show();
# print 'groundturth\n',solver.net.blobs['label'].data[:100]
# plt.imshow(solver.test_nets[0].blobs['data'].data[:8,0].transpose(1, 0, 2).reshape(28,8*28),cmap="gray");plt.axis('off');plt.show();
# print 'labels\n',solver.test_nets[0].blobs['label'].data[:8]

solver.solve()