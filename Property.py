from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Property:
    """Property type"""

    category: str
    name: str
    value: str

    def __init__(self, category: str, name: str, value: str):
        self.category = category
        self.name = name
        self.value = value

    def __str__(self) -> str:
        return f"category : {self.category} name : {self.name} value : {self.value}"
    
    def normalize(self) -> "Property":
        return Property(
            self.category.lower(),
            self.name.lower(),
            self.value.lower()
        )

    def get_text(self) -> str:
        return f"{self.category} {self.name} {self.value}".lower()

    def vectorize(self, vectorizer: TfidfVectorizer) -> np.ndarray:
        return vectorizer.transform([self.get_text()]).toarray()[0]
