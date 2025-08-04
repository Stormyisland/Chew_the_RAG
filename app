import tkinter as tk
from tkinter import filedialog, scrolledtext
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import to_rgba
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from transformers import BertModel, BertTokenizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class RAGEmbeddingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Word Embedding Visualizer")
        self.root.geometry("1200x800")
        
        # Initialize BERT model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        
        # GUI Components
        self.create_widgets()
        self.current_embeddings = None
        self.current_words = []
        
    def create_widgets(self):
        # Create frames
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        text_frame = tk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Control buttons
        tk.Button(control_frame, text="Load Document", command=self.load_document).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Generate Embeddings", command=self.generate_embeddings).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # Text areas
        self.original_text = scrolledtext.ScrolledText(text_frame, height=10, wrap=tk.WORD)
        self.original_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.original_text.insert(tk.END, "Original document will appear here...")
        
        self.processed_text = scrolledtext.ScrolledText(text_frame, height=8, wrap=tk.WORD)
        self.processed_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.processed_text.insert(tk.END, "Processed tokens will appear here...")
        
        # Visualization frame
        vis_frame = tk.Frame(self.root)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Connect click event
        self.canvas.mpl_connect('pick_event', self.on_plot_click)
        
    def load_document(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.original_text.delete(1.0, tk.END)
            self.original_text.insert(tk.END, content)
            self.processed_text.delete(1.0, tk.END)
            self.status.config(text=f"Loaded: {file_path}")
            
        except Exception as e:
            self.status.config(text=f"Error: {str(e)}")
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters/numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Remove short words
        tokens = [word for word in tokens if len(word) > 2]
        
        return tokens
    
    def generate_embeddings(self):
        text = self.original_text.get(1.0, tk.END)
        if not text.strip():
            self.status.config(text="No document loaded!")
            return
            
        try:
            # Preprocess text
            tokens = self.preprocess_text(text)
            unique_words = list(set(tokens))
            
            if not unique_words:
                self.status.config(text="No valid words after preprocessing!")
                return
                
            # Update processed text display
            self.processed_text.delete(1.0, tk.END)
            self.processed_text.insert(tk.END, " ".join(tokens))
            
            # Generate embeddings
            self.status.config(text="Generating embeddings...")
            self.root.update()
            
            embeddings = []
            valid_words = []
            
            for word in unique_words:
                inputs = self.tokenizer(word, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                if embedding.shape == (768,):  # BERT-base embedding size
                    embeddings.append(embedding)
                    valid_words.append(word)
            
            if not embeddings:
                self.status.config(text="No valid embeddings generated!")
                return
                
            self.current_embeddings = np.array(embeddings)
            self.current_words = valid_words
            
            # Reduce dimensionality with PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(self.current_embeddings)
            
            # Plot embeddings
            self.ax.clear()
            self.ax.set_title("Word Embedding Visualization (PCA Reduced)")
            self.ax.set_xlabel("Principal Component 1")
            self.ax.set_ylabel("Principal Component 2")
            
            # Create scatter plot with picker enabled
            scatter = self.ax.scatter(
                reduced_embeddings[:, 0], 
                reduced_embeddings[:, 1],
                alpha=0.6,
                picker=True  # Enable point picking
            )
            
            # Add labels for some points (not all to avoid clutter)
            for i, (x, y) in enumerate(reduced_embeddings):
                if i % 5 == 0:  # Label every 5th word
                    self.ax.annotate(
                        self.current_words[i], 
                        (x, y),
                        fontsize=8,
                        alpha=0.7
                    )
            
            self.canvas.draw()
            self.status.config(text=f"Generated embeddings for {len(valid_words)} words")
            
        except Exception as e:
            self.status.config(text=f"Error: {str(e)}")
    
    def on_plot_click(self, event):
        if not event.artist or not self.current_embeddings.any():
            return
            
        # Get the index of the clicked point
        ind = event.ind[0]
        word = self.current_words[ind]
        
        # Highlight selected word
        self.ax.clear()
        
        # Recreate the plot
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(self.current_embeddings)
        
        # Create color array (highlight selected point)
        colors = ['blue'] * len(self.current_words)
        colors[ind] = 'red'
        
        self.ax.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1],
            c=colors,
            alpha=0.6,
            picker=True
        )
        
        # Add labels
        for i, (x, y) in enumerate(reduced_embeddings):
            if i % 5 == 0 or i == ind:
                self.ax.annotate(
                    self.current_words[i], 
                    (x, y),
                    fontsize=8,
                    alpha=0.7
                )
        
        self.canvas.draw()
        self.status.config(text=f"Selected: '{word}' | Embedding size: 768 dimensions")
    
    def clear_all(self):
        self.original_text.delete(1.0, tk.END)
        self.processed_text.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()
        self.current_embeddings = None
        self.current_words = []
        self.status.config(text="Cleared all content")
self.

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGEmbeddingVisualizer(root)
    root.mainloop()
