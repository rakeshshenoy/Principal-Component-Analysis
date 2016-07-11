load('x_train.mat')
load('x_test.mat')
load('y_train.mat')
load('y_test.mat')

eigenvectors_train = get_sorted_eigenvecs(x_train);
eigenvectors_test = get_sorted_eigenvecs(x_test);

for i = 1:8
    img = 256 * double(reshape(eigenvectors(:,i), 16, 16));
    imwrite(img, sprintf('eig-%d.png', i));
end

K = [1, 3, 5, 15, 100];
samples = [250, 300, 450, 500, 3000];
for i = 1:numel(samples)
    n = samples(i);
    original = double(reshape(x_train(n,:), 16, 16));
    imwrite(original, sprintf('3c-orig-%d.png', n));
    for j = 1:numel(K)
        k = K(j);
        X_compressed = x_train * eigenvectors_train(:,1:k);
        X_reconstruction = X_compressed * eigenvectors_train(:,1:k)';
        y = double(reshape(X_reconstruction(n,:), 16, 16));
        imwrite(y, sprintf('3c-%d-%d.png', n, k));
        end
end

for i = 1:numel(K)
    k = K(i);
    X_compressed_train = x_train * eigenvectors_train(:,1:k);
    X_reconstruction_train = X_compressed_train * eigenvectors_train(:,1:k)';
    
    X_compressed_test = x_test * eigenvectors_test(:,1:k);
    X_reconstruction_test = X_compressed_test * eigenvectors_test(:,1:k)';

    tic;
    tree = ClassificationTree.fit(X_reconstruction_train, y_train, 'SplitCriterion', 'deviance');
    train_label_inferred = predict(tree, X_reconstruction_train);
    test_label_inferred = predict(tree, X_reconstruction_test);
    trainAccuracy = sum(train_label_inferred == y_train)/numel(y_train);
    testAccuracy = sum(test_label_inferred == y_test)/numel(y_test);
    msg = sprintf('#PC = %d\t Training Accuracy = %f\t Testing Accuracy = %f\t Time = %f', k, trainAccuracy, testAccuracy, toc);
    disp(msg);
end

