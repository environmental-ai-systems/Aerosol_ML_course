# Machine Learning for Aerosol Scientists

A hands-on, one-day introduction to machine learning for applied aerosol scientists. This course uses synthetic UVLIF (Ultraviolet Laser-Induced Fluorescence) mass spectra to teach fundamental ML concepts through practical examples.

## 🎯 Course Overview

This course is designed for aerosol scientists with little to no machine learning background. Through two interactive Jupyter notebooks, you'll learn:

- **Core ML concepts**: Train/test/validation splits, overfitting, hyperparameter tuning
- **Model evaluation**: ROC curves, AUC, confusion matrices, classification metrics
- **Two approaches**: Traditional ML (XGBoost) vs Deep Neural Networks
- **Practical skills**: Building, training, and evaluating models on spectroscopic data

## 📚 Notebooks

### 1. XGBoost for Aerosol Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebook1_xgboost_aerosol_classification.ipynb)

**Topics covered:**
- Data splitting strategies (train/test/validation)
- Building a baseline XGBoost model
- Understanding ROC curves and AUC
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Final model evaluation

**Duration:** ~2-3 hours

### 2. Deep Neural Networks for Aerosol Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebook2_dnn_aerosol_classification.ipynb)

**Topics covered:**
- Neural network architecture design
- Forward propagation and backpropagation
- Training dynamics and learning curves
- Regularization techniques (dropout, batch normalization)
- Comparing simple vs deep networks
- Visualizing what networks learn
- XGBoost vs DNN comparison

**Duration:** ~2-3 hours

## 🚀 Getting Started

### Prerequisites
- Google account (for Google Colab - free!)
- Basic Python knowledge (helpful but not required)
- No local installation needed!

### How to Use These Notebooks

1. **Click the "Open in Colab" badge** above for the notebook you want to start with
2. **Make a copy**: File → Save a copy in Drive (this saves your work)
3. **Run cells sequentially**: Press Shift+Enter to run each cell
4. **Experiment**: Change parameters, try different values, break things and fix them!

### First Time Using Colab?

Google Colab is a free Jupyter notebook environment that runs in your browser. No setup required!

- **Running code**: Click the play button or press Shift+Enter
- **Adding cells**: Click "+ Code" or "+ Text" 
- **Your work auto-saves** to your Google Drive
- **Free GPU access** (though not needed for these notebooks)

## 📊 Dataset

The notebooks use **synthetic UVLIF aerosol mass spectra** for four aerosol types:

1. **Biological** - Two fluorescence peaks (proteins and NAD(P)H)
2. **Mineral Dust** - Weak, broad fluorescence
3. **Organic Carbon** - Single moderate peak
4. **PAH** (Polycyclic Aromatic Hydrocarbons) - Strong peak at longer wavelengths

Each class has 250 samples with 64 wavelength channels. The synthetic data is generated with realistic noise and spectral characteristics.

## 🎓 Learning Outcomes

By the end of this course, you will be able to:

- [ ] Split data appropriately for model training and evaluation
- [ ] Train and evaluate machine learning models
- [ ] Interpret ROC curves, confusion matrices, and classification reports
- [ ] Perform hyperparameter tuning to optimize models
- [ ] Understand when to use XGBoost vs neural networks
- [ ] Apply these techniques to your own aerosol data

## 🔧 Course Structure

This is designed as a **one-day intensive course**:

| Time | Activity |
|------|----------|
| Morning Session (3h) | Notebook 1: XGBoost fundamentals |
| Lunch Break (1h) | Discussion and questions |
| Afternoon Session (3h) | Notebook 2: Deep neural networks |
| Wrap-up (30min) | Comparison, Q&A, next steps |

## 💡 Tips for Success

1. **Run every cell** - Don't skip ahead, each cell builds on previous ones
2. **Read the markdown** - Explanations are as important as code
3. **Experiment** - Try changing parameters and see what happens
4. **Ask questions** - Use comments in Colab or discussions in the course
5. **Compare results** - Note how different approaches perform

## 🛠️ Technical Details

### Libraries Used
- **scikit-learn**: ML utilities, preprocessing, metrics
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Deep learning framework
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

All libraries are pre-installed in Google Colab!

## 📖 Additional Resources

### Further Reading
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow/Keras Tutorials](https://www.tensorflow.org/tutorials)
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

### Aerosol Science + ML
- Consider exploring these techniques with your own UVLIF, ATOFMS, or AMS data
- Look into transfer learning for related classification tasks
- Explore dimensionality reduction (PCA, t-SNE) for exploratory analysis

## 🤝 Contributing

Found a bug or have suggestions? Please:
1. Open an issue on GitHub
2. Submit a pull request
3. Share your improvements!

## 📝 License

This educational material is released under the MIT License. Feel free to use, modify, and distribute for educational purposes.

## 👥 Authors & Acknowledgments

Created for applied aerosol scientists learning machine learning.

**Acknowledgments:**
- Thanks to the aerosol science community for inspiring this course
- Built using open-source tools and libraries

## 📧 Contact

Questions about the course? [Open an issue](https://github.com/YOUR_USERNAME/YOUR_REPO/issues) on GitHub.

---

## 🚦 Quick Start Checklist

- [ ] Click "Open in Colab" badge for Notebook 1
- [ ] Save a copy to your Google Drive
- [ ] Run the first cell to install packages
- [ ] Follow along with the explanations
- [ ] Complete exercises and experiments
- [ ] Move on to Notebook 2
- [ ] Compare your results!

**Ready to start?** Click the badge above and dive in! 🎉

---

*Last updated: February 2026*
