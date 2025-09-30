import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    # Load image and convert to grayscale if needed
    img = Image.open(image_path).convert('L')
    # Resize to a smaller size to reduce computation time
    img = img.resize((128, 128))  # Reduced from 224x224 to 64x64
    # Convert to numpy array and flatten
    return np.array(img).flatten()

def load_dataset(base_path):
    features = []
    labels = []
    
    # Load fake spectrograms
    fake_path = os.path.join(base_path, 'fake')
    for img_name in os.listdir(fake_path):
        if img_name.endswith('.png'):
            img_path = os.path.join(fake_path, img_name)
            features.append(load_and_preprocess_image(img_path))
            labels.append(0)  # 0 for fake
            
    # Load real spectrograms
    real_path = os.path.join(base_path, 'real')
    for img_name in os.listdir(real_path):
        if img_name.endswith('.png'):
            img_path = os.path.join(real_path, img_name)
            features.append(load_and_preprocess_image(img_path))
            labels.append(1)  # 1 for real
            
    return np.array(features), np.array(labels)

def train_svm_classifier(X, y):
    # Use StratifiedKFold for cross-validation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lists to store scores
    cv_scores = []
    
    print("\nPerforming 5-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train SVM
        print(f"\nTraining fold {fold}/5...")
        svm = SVC(kernel='linear', C=1.0, random_state=42, max_iter=10000, verbose=True)
        svm.fit(X_train, y_train)
        
        # Evaluate
        score = svm.score(X_test, y_test)
        cv_scores.append(score)
        print(f"Fold {fold} accuracy: {score:.4f}")
        
        # For the last fold, get detailed metrics
        if fold == 5:
            y_pred = svm.predict(X_test)
            print("\nFinal fold classification report:")
            print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
            
            # Plot confusion matrix for the last fold
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix (Last Fold)')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Fake', 'Real'])
            plt.yticks(tick_marks, ['Fake', 'Real'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Add numbers to the confusion matrix
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            plt.close()
    
    # Print cross-validation results
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
    
    # Train final model on full dataset for future use
    final_svm = SVC(kernel='linear', C=1.0, random_state=42, max_iter=1000)
    final_svm.fit(X_scaled, y)
    
    return final_svm, scaler
    
    # Train SVM
    print("Training SVM classifier...")
    svm = SVC(kernel='linear', C=1.0, random_state=42, max_iter=1000, verbose=True)  # Using linear kernel for faster training
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = svm.score(X_train_scaled, y_train)
    test_score = svm.score(X_test_scaled, y_test)
    
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Detailed classification report
    y_pred = svm.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Fake', 'Real'])
    plt.yticks(tick_marks, ['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add numbers to the confusion matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return svm, scaler

def evaluate_on_test_set(model, scaler, test_base_path):
    print("\nEvaluating on test set...")
    X_test, y_test = load_dataset(test_base_path)
    
    print(f"\nTest set summary:")
    print(f"Total test samples: {len(X_test)}")
    print(f"Fake test samples: {np.sum(y_test == 0)}")
    print(f"Real test samples: {np.sum(y_test == 1)}")
    
    # Scale the test features using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Plot confusion matrix for test set
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Set Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Fake', 'Real'])
    plt.yticks(tick_marks, ['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add numbers to the confusion matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png')
    plt.close()

def main():
    # Path to your spectrograms
    train_base_path = 'spectrograms_output'
    test_base_path = 'spectrograms_output_test'
    
    print("Loading and preprocessing training spectrograms...")
    X, y = load_dataset(train_base_path)
    
    print(f"\nTraining dataset summary:")
    print(f"Total samples: {len(X)}")
    print(f"Fake samples: {np.sum(y == 0)}")
    print(f"Real samples: {np.sum(y == 1)}")
    print(f"Feature vector shape: {X.shape}")
    
    # Train and evaluate the classifier
    svm_classifier, scaler = train_svm_classifier(X, y)
    
    # Evaluate on test set
    evaluate_on_test_set(svm_classifier, scaler, test_base_path)
    
    print("\nResults saved! Check confusion_matrix.png and test_confusion_matrix.png for the visualizations.")

if __name__ == "__main__":
    main()
