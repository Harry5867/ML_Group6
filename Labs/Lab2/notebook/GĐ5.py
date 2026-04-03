import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        n_estimators: số lượng cây
        learning_rate: tốc độ học
        max_depth: độ sâu tối đa của mỗi cây
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        Huấn luyện mô hình
        """
        # 1. Khởi tạo dự đoán ban đầu
        self.initial_prediction = np.mean(y)
        
        # Dự đoán ban đầu cho toàn bộ data
        y_pred = np.full(len(y), self.initial_prediction)

        # 2. Huấn luyện từng cây
        for i in range(self.n_estimators):
            # Tính residuals
            residuals = y - y_pred

            # Train cây mới trên residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Dự đoán từ cây mới
            tree_pred = tree.predict(X)

            # Cập nhật dự đoán
            y_pred += self.learning_rate * tree_pred

            # Lưu cây
            self.trees.append(tree)

            # Debug (tuỳ chọn)
            print(f"Tree {i+1}/{self.n_estimators} trained")

    def predict(self, X):
        """
        Dự đoán dữ liệu mới
        """
        # Bắt đầu từ giá trị trung bình
        y_pred = np.full(X.shape[0], self.initial_prediction)

        # Cộng dồn dự đoán từ các cây
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred
