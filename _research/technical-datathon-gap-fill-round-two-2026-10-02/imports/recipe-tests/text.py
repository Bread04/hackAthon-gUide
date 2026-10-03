import numpy as np
rng=np.random.default_rng(0)
pos=["great product works well","love it fast delivery","excellent quality happy"]; neg=["terrible broke quickly","awful support refund","bad quality never again"]
texts=[rng.choice(pos)+" "+str(i%7) if i%2 else rng.choice(neg)+" "+str(i%7) for i in range(200)]; y=np.array([i%2 for i in range(200)])
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    LogisticRegression(C=4.0, max_iter=2000, class_weight='balanced'))
print(cross_val_score(clf, texts, y, cv=5, scoring='f1_macro').mean())
