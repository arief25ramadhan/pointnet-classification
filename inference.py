# Load model
pointnet = POINTNET_MODEL()
model = pointnet.get_model()
model_weights_path = 'model.weights.h5'
model.load_weights(filemodel_weights_pathpath)


test_path = '/root/.keras/datasets/ModelNet10/toilet/test/toilet_0428.off'
points = np.array([trimesh.load(test_path).sample(2048)])

# # run test data through model
preds = model_loaded.predict(points)
print(preds)
preds = ops.argmax(preds, -1)

print(points)
print(preds)
print(CLASS_MAP[preds[0].numpy()])

# # plot points with predicted class and label
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")
ax.scatter(points[0, :, 0], points[0, :, 1], points[0, :, 2])
ax.set_title("pred: {:}".format(
    CLASS_MAP[preds[0].numpy()]
    )
)
ax.set_axis_off()
plt.show()