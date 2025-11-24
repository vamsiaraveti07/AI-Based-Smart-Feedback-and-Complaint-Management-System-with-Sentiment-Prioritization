"""
Root Cause Clustering System
Uses unsupervised AI to group similar grievances and detect patterns
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os
import logging
from collections import Counter, defaultdict
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class RootCauseAnalyzer:
    """
    Advanced root cause analysis using multiple clustering and topic modeling techniques
    """
    
    def __init__(self, db_path: str = 'grievance_system.db'):
        self.db_path = db_path
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['student', 'students', 'college', 'university', 'school'])
        
        # Initialize models
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.clustering_models = {}
        
        # Analysis results storage
        self.cluster_results = {}
        self.topic_models = {}
        self.root_causes = {}
        self.trend_analysis = {}
        
        # Initialize components
        self._init_models()
        self._create_analysis_tables()
    
    def _init_models(self):
        """Initialize machine learning models"""
        try:
            # Sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
    
    def _create_analysis_tables(self):
        """Create database tables for storing analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clusters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grievance_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grievance_id INTEGER,
                cluster_id INTEGER,
                cluster_label TEXT,
                cluster_method TEXT,
                similarity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (grievance_id) REFERENCES grievances (id)
            )
        ''')
        
        # Root causes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS root_causes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER,
                root_cause TEXT,
                description TEXT,
                frequency INTEGER,
                severity_score REAL,
                affected_departments TEXT,
                keywords TEXT,
                recommendations TEXT,
                status TEXT DEFAULT 'identified',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trend analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trend_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_period TEXT,
                category TEXT,
                department TEXT,
                trend_type TEXT,
                trend_value REAL,
                analysis_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Topic models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                topic_id INTEGER,
                topic_label TEXT,
                keywords TEXT,
                coherence_score REAL,
                documents_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def load_grievance_data(self, time_period: Optional[int] = None) -> pd.DataFrame:
        """Load grievance data for analysis"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT g.id, g.title, g.description, g.category, g.sentiment, 
                   g.priority, g.status, g.created_at, u.username
            FROM grievances g
            JOIN users u ON g.user_id = u.id
        '''
        
        params = []
        if time_period:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            query += ' WHERE g.created_at >= ?'
            params.append(cutoff_date.strftime('%Y-%m-%d %H:%M:%S'))
        
        query += ' ORDER BY g.created_at DESC'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Preprocess text
        df['processed_text'] = df['description'].apply(self.preprocess_text)
        df['combined_text'] = df['title'] + ' ' + df['processed_text']
        
        return df
    
    def perform_clustering_analysis(self, df: pd.DataFrame, methods: List[str] = None) -> Dict[str, Any]:
        """Perform multiple clustering analyses"""
        if methods is None:
            methods = ['kmeans', 'dbscan', 'hierarchical', 'hdbscan']
        
        results = {}
        
        # Prepare feature matrices
        feature_matrices = self._prepare_feature_matrices(df)
        
        for method in methods:
            try:
                if method == 'kmeans':
                    results[method] = self._kmeans_clustering(feature_matrices, df)
                elif method == 'dbscan':
                    results[method] = self._dbscan_clustering(feature_matrices, df)
                elif method == 'hierarchical':
                    results[method] = self._hierarchical_clustering(feature_matrices, df)
                elif method == 'hdbscan':
                    results[method] = self._hdbscan_clustering(feature_matrices, df)
                
                logger.info(f"Completed {method} clustering")
                
            except Exception as e:
                logger.error(f"Error in {method} clustering: {e}")
                results[method] = None
        
        # Store clustering results
        self.cluster_results = results
        self._save_clustering_results(df, results)
        
        return results
    
    def _prepare_feature_matrices(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare different feature matrices for clustering"""
        matrices = {}
        
        # TF-IDF features
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['combined_text'])
            matrices['tfidf'] = tfidf_matrix.toarray()
        except Exception as e:
            logger.error(f"Error creating TF-IDF matrix: {e}")
        
        # Sentence embeddings
        if self.sentence_transformer:
            try:
                embeddings = self.sentence_transformer.encode(df['combined_text'].tolist())
                matrices['embeddings'] = embeddings
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
        
        # Combined categorical + text features
        try:
            # Create category and sentiment dummy variables
            categorical_features = pd.get_dummies(df[['category', 'sentiment', 'priority']])
            
            if 'tfidf' in matrices:
                # Combine TF-IDF with categorical features
                combined = np.hstack([matrices['tfidf'], categorical_features.values])
                matrices['combined'] = combined
        except Exception as e:
            logger.error(f"Error creating combined features: {e}")
        
        return matrices
    
    def _kmeans_clustering(self, matrices: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform K-means clustering with optimal k selection"""
        best_results = {}
        
        for matrix_type, matrix in matrices.items():
            if matrix is None:
                continue
                
            # Find optimal number of clusters
            k_range = range(2, min(15, len(df) // 3))
            silhouette_scores = []
            inertias = []
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(matrix)
                    
                    if len(set(labels)) > 1:  # Ensure we have multiple clusters
                        silhouette_avg = silhouette_score(matrix, labels)
                        silhouette_scores.append(silhouette_avg)
                        inertias.append(kmeans.inertia_)
                    else:
                        silhouette_scores.append(-1)
                        inertias.append(float('inf'))
                        
                except Exception as e:
                    logger.error(f"Error in K-means for k={k}: {e}")
                    silhouette_scores.append(-1)
                    inertias.append(float('inf'))
            
            # Select optimal k
            if silhouette_scores:
                best_k = k_range[np.argmax(silhouette_scores)]
                
                # Final clustering with optimal k
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(matrix)
                
                best_results[matrix_type] = {
                    'model': kmeans,
                    'labels': labels,
                    'k': best_k,
                    'silhouette_score': max(silhouette_scores),
                    'inertia': kmeans.inertia_,
                    'cluster_centers': kmeans.cluster_centers_
                }
        
        return best_results
    
    def _dbscan_clustering(self, matrices: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform DBSCAN clustering"""
        results = {}
        
        for matrix_type, matrix in matrices.items():
            if matrix is None:
                continue
                
            # Try different eps values
            eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
            best_score = -1
            best_result = None
            
            for eps in eps_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=3)
                    labels = dbscan.fit_predict(matrix)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if n_clusters > 1:
                        # Calculate silhouette score (excluding noise points)
                        valid_indices = labels != -1
                        if np.sum(valid_indices) > 1:
                            score = silhouette_score(matrix[valid_indices], labels[valid_indices])
                            
                            if score > best_score:
                                best_score = score
                                best_result = {
                                    'model': dbscan,
                                    'labels': labels,
                                    'eps': eps,
                                    'n_clusters': n_clusters,
                                    'silhouette_score': score,
                                    'noise_points': np.sum(labels == -1)
                                }
                except Exception as e:
                    logger.error(f"Error in DBSCAN with eps={eps}: {e}")
            
            if best_result:
                results[matrix_type] = best_result
        
        return results
    
    def _hierarchical_clustering(self, matrices: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform hierarchical clustering"""
        results = {}
        
        for matrix_type, matrix in matrices.items():
            if matrix is None:
                continue
                
            # Try different numbers of clusters
            n_clusters_range = range(2, min(10, len(df) // 3))
            best_score = -1
            best_result = None
            
            for n_clusters in n_clusters_range:
                try:
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = hierarchical.fit_predict(matrix)
                    
                    score = silhouette_score(matrix, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'model': hierarchical,
                            'labels': labels,
                            'n_clusters': n_clusters,
                            'silhouette_score': score
                        }
                        
                except Exception as e:
                    logger.error(f"Error in hierarchical clustering with n_clusters={n_clusters}: {e}")
            
            if best_result:
                results[matrix_type] = best_result
        
        return results
    
    def _hdbscan_clustering(self, matrices: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform HDBSCAN clustering"""
        results = {}
        
        for matrix_type, matrix in matrices.items():
            if matrix is None:
                continue
                
            try:
                # Use UMAP for dimensionality reduction if needed
                if matrix.shape[1] > 50:
                    umap_model = umap.UMAP(n_components=10, random_state=42)
                    matrix_reduced = umap_model.fit_transform(matrix)
                else:
                    matrix_reduced = matrix
                
                # HDBSCAN clustering
                clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
                labels = clusterer.fit_predict(matrix_reduced)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    # Calculate silhouette score (excluding noise points)
                    valid_indices = labels != -1
                    if np.sum(valid_indices) > 1:
                        score = silhouette_score(matrix_reduced[valid_indices], labels[valid_indices])
                        
                        results[matrix_type] = {
                            'model': clusterer,
                            'labels': labels,
                            'n_clusters': n_clusters,
                            'silhouette_score': score,
                            'noise_points': np.sum(labels == -1),
                            'cluster_persistence': clusterer.cluster_persistence_
                        }
                        
            except Exception as e:
                logger.error(f"Error in HDBSCAN clustering: {e}")
        
        return results
    
    def perform_topic_modeling(self, df: pd.DataFrame, methods: List[str] = None) -> Dict[str, Any]:
        """Perform topic modeling analysis"""
        if methods is None:
            methods = ['lda', 'nmf']
        
        results = {}
        
        # Prepare documents
        documents = df['combined_text'].tolist()
        
        for method in methods:
            try:
                if method == 'lda':
                    results[method] = self._lda_topic_modeling(documents)
                elif method == 'nmf':
                    results[method] = self._nmf_topic_modeling(documents)
                    
                logger.info(f"Completed {method} topic modeling")
                
            except Exception as e:
                logger.error(f"Error in {method} topic modeling: {e}")
                results[method] = None
        
        self.topic_models = results
        self._save_topic_models(results)
        
        return results
    
    def _lda_topic_modeling(self, documents: List[str]) -> Dict[str, Any]:
        """Perform LDA topic modeling"""
        # Tokenize documents
        tokenized_docs = [doc.split() for doc in documents]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        # Find optimal number of topics
        topic_range = range(2, min(10, len(documents) // 3))
        coherence_scores = []
        
        for num_topics in topic_range:
            try:
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=42,
                    passes=10,
                    alpha='auto',
                    per_word_topics=True
                )
                
                # Calculate coherence score
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                
                coherence_score = coherence_model.get_coherence()
                coherence_scores.append(coherence_score)
                
            except Exception as e:
                logger.error(f"Error in LDA with {num_topics} topics: {e}")
                coherence_scores.append(0)
        
        # Select optimal number of topics
        if coherence_scores:
            best_num_topics = topic_range[np.argmax(coherence_scores)]
            
            # Final LDA model
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=best_num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            # Extract topics
            topics = []
            for i in range(best_num_topics):
                topic_words = lda_model.show_topic(i, topn=10)
                topics.append({
                    'id': i,
                    'words': topic_words,
                    'label': self._generate_topic_label([word for word, _ in topic_words])
                })
            
            return {
                'model': lda_model,
                'dictionary': dictionary,
                'corpus': corpus,
                'topics': topics,
                'num_topics': best_num_topics,
                'coherence_score': max(coherence_scores)
            }
        
        return None
    
    def _nmf_topic_modeling(self, documents: List[str]) -> Dict[str, Any]:
        """Perform NMF topic modeling"""
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        # Find optimal number of topics
        topic_range = range(2, min(10, len(documents) // 3))
        reconstruction_errors = []
        
        for num_topics in topic_range:
            try:
                nmf_model = NMF(
                    n_components=num_topics,
                    random_state=42,
                    alpha_W=0.1,
                    alpha_H=0.1,
                    l1_ratio=0.5,
                    max_iter=200
                )
                
                nmf_model.fit(tfidf_matrix)
                reconstruction_errors.append(nmf_model.reconstruction_err_)
                
            except Exception as e:
                logger.error(f"Error in NMF with {num_topics} topics: {e}")
                reconstruction_errors.append(float('inf'))
        
        # Select optimal number of topics (elbow method)
        if reconstruction_errors:
            # Use elbow method or select middle value
            best_num_topics = topic_range[len(topic_range) // 2]
            
            # Final NMF model
            nmf_model = NMF(
                n_components=best_num_topics,
                random_state=42,
                alpha_W=0.1,
                alpha_H=0.1,
                l1_ratio=0.5,
                max_iter=200
            )
            
            W = nmf_model.fit_transform(tfidf_matrix)
            H = nmf_model.components_
            
            # Extract topics
            topics = []
            for i in range(best_num_topics):
                top_indices = H[i].argsort()[-10:][::-1]
                topic_words = [(feature_names[idx], H[i][idx]) for idx in top_indices]
                topics.append({
                    'id': i,
                    'words': topic_words,
                    'label': self._generate_topic_label([word for word, _ in topic_words])
                })
            
            return {
                'model': nmf_model,
                'vectorizer': vectorizer,
                'topics': topics,
                'num_topics': best_num_topics,
                'reconstruction_error': nmf_model.reconstruction_err_
            }
        
        return None
    
    def _generate_topic_label(self, words: List[str]) -> str:
        """Generate a human-readable label for a topic"""
        # Use the top 3 words to create a label
        top_words = words[:3]
        return '_'.join(top_words).replace(' ', '_')
    
    def identify_root_causes(self, df: pd.DataFrame, cluster_results: Dict, 
                           topic_results: Dict) -> Dict[str, Any]:
        """Identify root causes from clustering and topic modeling results"""
        root_causes = {}
        
        # Process each clustering method
        for method_name, method_results in cluster_results.items():
            if method_results is None:
                continue
                
            for matrix_type, result in method_results.items():
                if result is None:
                    continue
                    
                labels = result['labels']
                unique_labels = set(labels)
                
                # Remove noise label (-1) if present
                if -1 in unique_labels:
                    unique_labels.remove(-1)
                
                cluster_analysis = {}
                
                for cluster_id in unique_labels:
                    cluster_mask = labels == cluster_id
                    cluster_data = df[cluster_mask]
                    
                    if len(cluster_data) < 2:  # Skip small clusters
                        continue
                    
                    # Analyze cluster characteristics
                    cluster_info = self._analyze_cluster(cluster_data, cluster_id)
                    cluster_analysis[cluster_id] = cluster_info
                
                root_causes[f"{method_name}_{matrix_type}"] = cluster_analysis
        
        self.root_causes = root_causes
        self._save_root_causes(root_causes)
        
        return root_causes
    
    def _analyze_cluster(self, cluster_data: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Analyze characteristics of a specific cluster"""
        analysis = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(cluster_data) * 100,
            'categories': {},
            'sentiments': {},
            'priorities': {},
            'keywords': [],
            'time_pattern': {},
            'root_cause_hypothesis': '',
            'severity_score': 0.0,
            'recommendations': []
        }
        
        # Category distribution
        category_counts = cluster_data['category'].value_counts()
        analysis['categories'] = category_counts.to_dict()
        
        # Sentiment distribution
        sentiment_counts = cluster_data['sentiment'].value_counts()
        analysis['sentiments'] = sentiment_counts.to_dict()
        
        # Priority distribution
        priority_counts = cluster_data['priority'].value_counts()
        analysis['priorities'] = priority_counts.to_dict()
        
        # Extract keywords using TF-IDF
        analysis['keywords'] = self._extract_cluster_keywords(cluster_data['combined_text'])
        
        # Time pattern analysis
        analysis['time_pattern'] = self._analyze_time_patterns(cluster_data)
        
        # Generate root cause hypothesis
        analysis['root_cause_hypothesis'] = self._generate_root_cause_hypothesis(analysis)
        
        # Calculate severity score
        analysis['severity_score'] = self._calculate_cluster_severity(analysis)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_cluster_recommendations(analysis)
        
        return analysis
    
    def _extract_cluster_keywords(self, texts: pd.Series, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract key terms from cluster texts"""
        try:
            # Create TF-IDF matrix for this cluster
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = mean_scores.argsort()[-top_n:][::-1]
            keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _analyze_time_patterns(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in cluster data"""
        time_analysis = {}
        
        try:
            # Convert created_at to datetime
            cluster_data = cluster_data.copy()
            cluster_data['created_at'] = pd.to_datetime(cluster_data['created_at'])
            
            # Extract time components
            cluster_data['hour'] = cluster_data['created_at'].dt.hour
            cluster_data['day_of_week'] = cluster_data['created_at'].dt.dayofweek
            cluster_data['month'] = cluster_data['created_at'].dt.month
            
            # Analyze patterns
            time_analysis['hourly_pattern'] = cluster_data['hour'].value_counts().to_dict()
            time_analysis['weekly_pattern'] = cluster_data['day_of_week'].value_counts().to_dict()
            time_analysis['monthly_pattern'] = cluster_data['month'].value_counts().to_dict()
            
            # Find peak times
            time_analysis['peak_hour'] = cluster_data['hour'].mode().iloc[0] if not cluster_data['hour'].mode().empty else None
            time_analysis['peak_day'] = cluster_data['day_of_week'].mode().iloc[0] if not cluster_data['day_of_week'].mode().empty else None
            
        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
        
        return time_analysis
    
    def _generate_root_cause_hypothesis(self, analysis: Dict) -> str:
        """Generate a hypothesis about the root cause"""
        hypotheses = []
        
        # Category-based hypothesis
        if analysis['categories']:
            dominant_category = max(analysis['categories'].items(), key=lambda x: x[1])[0]
            hypotheses.append(f"Issues primarily related to {dominant_category}")
        
        # Sentiment-based hypothesis
        if analysis['sentiments']:
            negative_ratio = analysis['sentiments'].get('negative', 0) / analysis['size']
            if negative_ratio > 0.7:
                hypotheses.append("High negative sentiment indicates systemic issues")
        
        # Priority-based hypothesis
        if analysis['priorities']:
            high_priority_ratio = analysis['priorities'].get(1, 0) / analysis['size']
            if high_priority_ratio > 0.5:
                hypotheses.append("Many high-priority cases suggest urgent systematic problems")
        
        # Keyword-based hypothesis
        if analysis['keywords']:
            top_keywords = [kw[0] for kw in analysis['keywords'][:3]]
            hypotheses.append(f"Key issues around: {', '.join(top_keywords)}")
        
        # Time-based hypothesis
        if analysis['time_pattern'].get('peak_hour') is not None:
            peak_hour = analysis['time_pattern']['peak_hour']
            if 9 <= peak_hour <= 17:
                hypotheses.append("Issues occur during business hours, suggesting operational problems")
            else:
                hypotheses.append("Issues occur outside business hours, suggesting system/facility problems")
        
        return " | ".join(hypotheses) if hypotheses else "No clear pattern identified"
    
    def _calculate_cluster_severity(self, analysis: Dict) -> float:
        """Calculate severity score for a cluster"""
        severity_factors = []
        
        # Size factor (larger clusters are more severe)
        size_factor = min(analysis['size'] / 10, 1.0)
        severity_factors.append(size_factor)
        
        # Sentiment factor
        if analysis['sentiments']:
            negative_ratio = analysis['sentiments'].get('negative', 0) / analysis['size']
            severity_factors.append(negative_ratio)
        
        # Priority factor
        if analysis['priorities']:
            high_priority_ratio = analysis['priorities'].get(1, 0) / analysis['size']
            severity_factors.append(high_priority_ratio)
        
        # Keyword severity (based on presence of urgent keywords)
        urgent_keywords = ['broken', 'emergency', 'urgent', 'critical', 'dangerous', 'unsafe']
        keyword_severity = 0
        if analysis['keywords']:
            for keyword, score in analysis['keywords']:
                if any(urgent in keyword.lower() for urgent in urgent_keywords):
                    keyword_severity += score
        severity_factors.append(min(keyword_severity, 1.0))
        
        return np.mean(severity_factors) if severity_factors else 0.0
    
    def _generate_cluster_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations for addressing cluster issues"""
        recommendations = []
        
        # Size-based recommendations
        if analysis['size'] >= 5:
            recommendations.append("HIGH PRIORITY: Large number of similar complaints - investigate systematic issues")
        
        # Category-based recommendations
        if analysis['categories']:
            dominant_category = max(analysis['categories'].items(), key=lambda x: x[1])[0]
            
            category_recommendations = {
                'hostel': "Review hostel facilities and management processes",
                'academic': "Examine academic policies and faculty performance",
                'infrastructure': "Assess IT and facility infrastructure",
                'administration': "Review administrative procedures and staff training"
            }
            
            if dominant_category in category_recommendations:
                recommendations.append(category_recommendations[dominant_category])
        
        # Sentiment-based recommendations
        if analysis['sentiments']:
            negative_ratio = analysis['sentiments'].get('negative', 0) / analysis['size']
            if negative_ratio > 0.8:
                recommendations.append("URGENT: Very high negative sentiment - immediate management attention required")
        
        # Time-based recommendations
        time_pattern = analysis.get('time_pattern', {})
        peak_hour = time_pattern.get('peak_hour')
        if peak_hour is not None:
            if 22 <= peak_hour or peak_hour <= 6:
                recommendations.append("Issues occurring at night - check facility security and maintenance")
            elif 12 <= peak_hour <= 14:
                recommendations.append("Issues during lunch hours - check service availability")
        
        # Keyword-based recommendations
        if analysis['keywords']:
            for keyword, score in analysis['keywords'][:5]:
                if 'wifi' in keyword or 'internet' in keyword:
                    recommendations.append("Network connectivity issues - audit IT infrastructure")
                elif 'food' in keyword or 'mess' in keyword:
                    recommendations.append("Food service issues - review catering and hygiene standards")
                elif 'room' in keyword or 'accommodation' in keyword:
                    recommendations.append("Accommodation issues - inspect housing facilities")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def perform_trend_analysis(self, time_period_days: int = 90) -> Dict[str, Any]:
        """Perform trend analysis over time"""
        # Load data for the specified period
        df = self.load_grievance_data(time_period_days)
        
        if len(df) < 10:
            return {"error": "Insufficient data for trend analysis"}
        
        # Convert dates
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        df['week'] = df['created_at'].dt.isocalendar().week
        df['month'] = df['created_at'].dt.month
        
        trend_analysis = {}
        
        # Overall volume trends
        trend_analysis['volume_trends'] = self._analyze_volume_trends(df)
        
        # Category trends
        trend_analysis['category_trends'] = self._analyze_category_trends(df)
        
        # Sentiment trends
        trend_analysis['sentiment_trends'] = self._analyze_sentiment_trends(df)
        
        # Department trends (if departments are tracked)
        trend_analysis['department_trends'] = self._analyze_department_trends(df)
        
        # Seasonal patterns
        trend_analysis['seasonal_patterns'] = self._analyze_seasonal_patterns(df)
        
        # Emerging issues
        trend_analysis['emerging_issues'] = self._identify_emerging_issues(df)
        
        self.trend_analysis = trend_analysis
        self._save_trend_analysis(trend_analysis)
        
        return trend_analysis
    
    def _analyze_volume_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze complaint volume trends"""
        daily_counts = df.groupby('date').size()
        weekly_counts = df.groupby('week').size()
        
        # Calculate trend direction
        if len(daily_counts) >= 7:
            recent_avg = daily_counts.tail(7).mean()
            previous_avg = daily_counts.head(7).mean() if len(daily_counts) >= 14 else recent_avg
            trend_direction = "increasing" if recent_avg > previous_avg * 1.1 else "decreasing" if recent_avg < previous_avg * 0.9 else "stable"
        else:
            trend_direction = "insufficient_data"
        
        return {
            'daily_average': daily_counts.mean(),
            'peak_day': daily_counts.idxmax() if not daily_counts.empty else None,
            'peak_count': daily_counts.max() if not daily_counts.empty else 0,
            'trend_direction': trend_direction,
            'weekly_pattern': weekly_counts.to_dict()
        }
    
    def _analyze_category_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends by category"""
        category_trends = {}
        
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            daily_counts = cat_data.groupby('date').size()
            
            if len(daily_counts) >= 3:
                recent_avg = daily_counts.tail(7).mean() if len(daily_counts) >= 7 else daily_counts.mean()
                total_count = len(cat_data)
                
                category_trends[category] = {
                    'total_complaints': total_count,
                    'daily_average': recent_avg,
                    'percentage_of_total': (total_count / len(df)) * 100,
                    'trend': self._calculate_trend_direction(daily_counts)
                }
        
        return category_trends
    
    def _analyze_sentiment_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment trends over time"""
        sentiment_by_date = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        sentiment_trends = {}
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_by_date.columns:
                sentiment_data = sentiment_by_date[sentiment]
                sentiment_trends[sentiment] = {
                    'average_daily': sentiment_data.mean(),
                    'trend': self._calculate_trend_direction(sentiment_data),
                    'percentage': (sentiment_data.sum() / len(df)) * 100
                }
        
        # Calculate sentiment deterioration
        if 'negative' in sentiment_trends and 'positive' in sentiment_trends:
            neg_trend = sentiment_trends['negative']['trend']
            pos_trend = sentiment_trends['positive']['trend']
            
            if neg_trend == 'increasing' and pos_trend == 'decreasing':
                sentiment_trends['overall_trend'] = 'deteriorating'
            elif neg_trend == 'decreasing' and pos_trend == 'increasing':
                sentiment_trends['overall_trend'] = 'improving'
            else:
                sentiment_trends['overall_trend'] = 'stable'
        
        return sentiment_trends
    
    def _analyze_department_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends by department (if available)"""
        # This would require department mapping based on categories
        department_mapping = {
            'academic': 'Academic Affairs',
            'hostel': 'Student Housing',
            'infrastructure': 'IT & Infrastructure',
            'administration': 'Administration'
        }
        
        df['department'] = df['category'].map(department_mapping).fillna('Other')
        
        dept_trends = {}
        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]
            daily_counts = dept_data.groupby('date').size()
            
            dept_trends[dept] = {
                'total_complaints': len(dept_data),
                'daily_average': daily_counts.mean(),
                'trend': self._calculate_trend_direction(daily_counts),
                'priority_distribution': dept_data['priority'].value_counts().to_dict()
            }
        
        return dept_trends
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal and temporal patterns"""
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.day_name()
        df['month'] = df['created_at'].dt.month_name()
        
        patterns = {
            'hourly_distribution': df.groupby('hour').size().to_dict(),
            'daily_distribution': df.groupby('day_of_week').size().to_dict(),
            'monthly_distribution': df.groupby('month').size().to_dict()
        }
        
        # Identify peak times
        patterns['peak_hour'] = df['hour'].mode().iloc[0] if not df['hour'].mode().empty else None
        patterns['peak_day'] = df['day_of_week'].mode().iloc[0] if not df['day_of_week'].mode().empty else None
        patterns['peak_month'] = df['month'].mode().iloc[0] if not df['month'].mode().empty else None
        
        return patterns
    
    def _identify_emerging_issues(self, df: pd.DataFrame, window_days: int = 14) -> List[Dict[str, Any]]:
        """Identify emerging issues using keyword frequency analysis"""
        # Split data into recent and historical periods
        cutoff_date = df['created_at'].max() - timedelta(days=window_days)
        recent_data = df[df['created_at'] >= cutoff_date]
        historical_data = df[df['created_at'] < cutoff_date]
        
        if len(recent_data) < 5 or len(historical_data) < 5:
            return []
        
        # Extract keywords from both periods
        recent_keywords = self._extract_period_keywords(recent_data['combined_text'])
        historical_keywords = self._extract_period_keywords(historical_data['combined_text'])
        
        # Find emerging keywords
        emerging_issues = []
        for keyword, recent_freq in recent_keywords.items():
            historical_freq = historical_keywords.get(keyword, 0)
            
            # Calculate growth rate
            if historical_freq > 0:
                growth_rate = (recent_freq - historical_freq) / historical_freq
            else:
                growth_rate = float('inf') if recent_freq > 0 else 0
            
            # Identify significant emergences
            if growth_rate > 2.0 and recent_freq >= 3:  # At least 200% growth and 3 occurrences
                emerging_issues.append({
                    'keyword': keyword,
                    'recent_frequency': recent_freq,
                    'historical_frequency': historical_freq,
                    'growth_rate': growth_rate,
                    'severity': 'high' if growth_rate > 5.0 else 'medium'
                })
        
        # Sort by growth rate
        emerging_issues.sort(key=lambda x: x['growth_rate'], reverse=True)
        
        return emerging_issues[:10]  # Top 10 emerging issues
    
    def _extract_period_keywords(self, texts: pd.Series) -> Dict[str, int]:
        """Extract keyword frequencies from a period of texts"""
        all_text = ' '.join(texts)
        words = all_text.split()
        
        # Filter meaningful words
        meaningful_words = [
            word for word in words 
            if len(word) > 3 and word not in self.stop_words
        ]
        
        return Counter(meaningful_words)
    
    def _calculate_trend_direction(self, time_series: pd.Series) -> str:
        """Calculate trend direction for a time series"""
        if len(time_series) < 3:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = np.arange(len(time_series))
        y = time_series.values
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.3:
            return "increasing"
        elif correlation < -0.3:
            return "decreasing"
        else:
            return "stable"
    
    def _save_clustering_results(self, df: pd.DataFrame, results: Dict):
        """Save clustering results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for method_name, method_results in results.items():
                if method_results is None:
                    continue
                    
                for matrix_type, result in method_results.items():
                    if result is None:
                        continue
                        
                    labels = result['labels']
                    
                    for idx, label in enumerate(labels):
                        if idx < len(df):
                            grievance_id = df.iloc[idx]['id']
                            
                            cursor.execute('''
                                INSERT OR REPLACE INTO grievance_clusters 
                                (grievance_id, cluster_id, cluster_label, cluster_method, similarity_score)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (
                                grievance_id,
                                int(label),
                                f"{method_name}_{matrix_type}_{label}",
                                f"{method_name}_{matrix_type}",
                                result.get('silhouette_score', 0.0)
                            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving clustering results: {e}")
    
    def _save_root_causes(self, root_causes: Dict):
        """Save root cause analysis to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for method_name, clusters in root_causes.items():
                for cluster_id, analysis in clusters.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO root_causes 
                        (cluster_id, root_cause, description, frequency, severity_score, 
                         keywords, recommendations)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        cluster_id,
                        analysis['root_cause_hypothesis'],
                        f"Cluster analysis for {method_name}",
                        analysis['size'],
                        analysis['severity_score'],
                        json.dumps(analysis['keywords']),
                        json.dumps(analysis['recommendations'])
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving root causes: {e}")
    
    def _save_topic_models(self, topic_results: Dict):
        """Save topic modeling results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for method_name, result in topic_results.items():
                if result is None:
                    continue
                    
                topics = result.get('topics', [])
                
                for topic in topics:
                    cursor.execute('''
                        INSERT OR REPLACE INTO topic_models 
                        (model_type, topic_id, topic_label, keywords, coherence_score, documents_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        method_name,
                        topic['id'],
                        topic['label'],
                        json.dumps(topic['words']),
                        result.get('coherence_score', 0.0),
                        0  # Would need document assignment for accurate count
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving topic models: {e}")
    
    def _save_trend_analysis(self, trend_analysis: Dict):
        """Save trend analysis to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = datetime.now().strftime('%Y-%m-%d')
            
            for trend_type, data in trend_analysis.items():
                cursor.execute('''
                    INSERT INTO trend_analysis 
                    (time_period, category, trend_type, analysis_data)
                    VALUES (?, ?, ?, ?)
                ''', (
                    current_time,
                    'overall',
                    trend_type,
                    json.dumps(data)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trend analysis: {e}")
    
    def generate_comprehensive_report(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Generate a comprehensive root cause analysis report"""
        # Load data
        df = self.load_grievance_data(time_period_days)
        
        if len(df) < 5:
            return {"error": "Insufficient data for analysis"}
        
        # Perform all analyses
        cluster_results = self.perform_clustering_analysis(df)
        topic_results = self.perform_topic_modeling(df)
        root_causes = self.identify_root_causes(df, cluster_results, topic_results)
        trend_analysis = self.perform_trend_analysis(time_period_days)
        
        # Generate summary
        report = {
            'analysis_period': f"{time_period_days} days",
            'total_grievances': len(df),
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': self._generate_analysis_summary(df, root_causes, trend_analysis),
            'cluster_analysis': cluster_results,
            'topic_analysis': topic_results,
            'root_causes': root_causes,
            'trend_analysis': trend_analysis,
            'recommendations': self._generate_overall_recommendations(root_causes, trend_analysis),
            'action_items': self._generate_action_items(root_causes, trend_analysis)
        }
        
        return report
    
    def _generate_analysis_summary(self, df: pd.DataFrame, root_causes: Dict, 
                                 trend_analysis: Dict) -> Dict[str, Any]:
        """Generate analysis summary"""
        summary = {
            'total_complaints': len(df),
            'categories_affected': df['category'].nunique(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'priority_distribution': df['priority'].value_counts().to_dict(),
        }
        
        # Find most common issues
        if root_causes:
            all_clusters = []
            for method_results in root_causes.values():
                all_clusters.extend(method_results.values())
            
            if all_clusters:
                largest_cluster = max(all_clusters, key=lambda x: x['size'])
                summary['largest_issue_cluster'] = {
                    'size': largest_cluster['size'],
                    'hypothesis': largest_cluster['root_cause_hypothesis'],
                    'severity': largest_cluster['severity_score']
                }
        
        # Trend summary
        if trend_analysis.get('volume_trends'):
            summary['volume_trend'] = trend_analysis['volume_trends']['trend_direction']
        
        if trend_analysis.get('sentiment_trends'):
            summary['sentiment_trend'] = trend_analysis['sentiment_trends'].get('overall_trend', 'unknown')
        
        return summary
    
    def _generate_overall_recommendations(self, root_causes: Dict, 
                                        trend_analysis: Dict) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []
        
        # From root causes
        if root_causes:
            high_severity_clusters = []
            for method_results in root_causes.values():
                for cluster in method_results.values():
                    if cluster['severity_score'] > 0.7:
                        high_severity_clusters.append(cluster)
            
            if high_severity_clusters:
                recommendations.append("URGENT: High-severity issue clusters identified requiring immediate attention")
                
                # Get specific recommendations
                for cluster in high_severity_clusters[:3]:  # Top 3
                    recommendations.extend(cluster['recommendations'][:2])  # Top 2 per cluster
        
        # From trend analysis
        if trend_analysis.get('volume_trends', {}).get('trend_direction') == 'increasing':
            recommendations.append("Volume trend increasing - consider expanding support capacity")
        
        if trend_analysis.get('sentiment_trends', {}).get('overall_trend') == 'deteriorating':
            recommendations.append("ALERT: Sentiment deteriorating - review service quality and response times")
        
        # From emerging issues
        emerging = trend_analysis.get('emerging_issues', [])
        if emerging:
            top_emerging = emerging[0]
            recommendations.append(f"Monitor emerging issue: {top_emerging['keyword']} (growth: {top_emerging['growth_rate']:.1f}x)")
        
        return recommendations[:10]  # Limit to top 10
    
    def _generate_action_items(self, root_causes: Dict, trend_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate specific action items"""
        action_items = []
        
        # High-priority clusters
        if root_causes:
            for method_name, method_results in root_causes.items():
                for cluster_id, cluster in method_results.items():
                    if cluster['severity_score'] > 0.6:
                        action_items.append({
                            'priority': 'high' if cluster['severity_score'] > 0.8 else 'medium',
                            'title': f"Address {cluster['root_cause_hypothesis']}",
                            'description': f"Cluster of {cluster['size']} complaints with severity {cluster['severity_score']:.2f}",
                            'recommendations': cluster['recommendations'][:3],
                            'affected_categories': list(cluster['categories'].keys()),
                            'due_date': 'immediate' if cluster['severity_score'] > 0.8 else '7_days'
                        })
        
        # Trend-based action items
        if trend_analysis.get('emerging_issues'):
            for issue in trend_analysis['emerging_issues'][:3]:
                action_items.append({
                    'priority': issue['severity'],
                    'title': f"Investigate emerging issue: {issue['keyword']}",
                    'description': f"New issue with {issue['growth_rate']:.1f}x growth rate",
                    'recommendations': [f"Research root cause of {issue['keyword']} complaints"],
                    'due_date': '3_days'
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        action_items.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return action_items[:15]  # Limit to top 15 action items
