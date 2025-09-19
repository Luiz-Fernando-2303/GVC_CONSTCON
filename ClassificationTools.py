from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from getItems import getItems, getItemsByFilter
from Property import Property
from ModelItem import ModelItem, ClassificationItem
import numpy as np
from typing import List
from tqdm import tqdm

def getClassificationItems(limit: int = 0, filter: dict[str, str] = {}, labelDefinition: List[dict[str, str]] = []) -> List[ClassificationItem]:
    if filter and limit > 0:
        items = getItemsByFilter(filter, limit)
    elif filter:
        items = getItemsByFilter(filter, 0)
    elif limit > 0:
        items = getItems(limit)
    else:
        items = getItems(0)

    # create classification items
    ClassificationItems: list[ClassificationItem] = []
    for item in items:
        if not labelDefinition:
            ClassificationItems.append(ClassificationItem("", item.properties))
        else:
            labels = []
            for label_def in labelDefinition:
                label_property = next(
                    (prop for prop in item.properties if prop.category == label_def.get("Category", "") and prop.name == label_def.get("Name", "")),
                    None
                )
                if label_property:
                    labels.append(f"{label_property.value}")
            if labels:
                available_properties = [prop for prop in item.properties if not any(lp.get("Category") == prop.category and lp.get("Name") == prop.name for lp in labelDefinition)]
                ClassificationItems.append(ClassificationItem(",".join(labels), available_properties))

    propertiesSamples = [p.properties for p in ClassificationItems] # samples of properties

    # create properties list for vectorization
    properties: list[Property] = []
    for sample in propertiesSamples:
        for prop in sample:
            properties.append(prop.normalize())

    # create global vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit([p.get_text() for p in properties])

    # vectorize the properties of each ClassificationItem
    finalItems: list[ClassificationItem] = []
    for item in tqdm(ClassificationItems, desc="Vectorizing properties"):
        
        item.vectorizer = vectorizer

        # vectorize properties
        item.vectorizedProperties = [prop.vectorize(vectorizer) for prop in item.properties]
        finalItems.append(item)

    return finalItems