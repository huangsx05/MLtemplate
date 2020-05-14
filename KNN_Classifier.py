# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
score = knn.score(X_test, y_test)

# plot decision boundry for k = 1, 3, 9
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	ax.set_title("{} neighbor(s)".format(n_neighbors))
	ax.set_xlabel("feature 0")
	ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()

# k sensitivity study - plot accuracy vs n_neighbor (k)
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(X_train, y_train)
	training_accuracy.append(clf.score(X_train, y_train))
	test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()