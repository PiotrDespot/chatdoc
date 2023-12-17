import io

import numpy as np

from enum import Enum
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Dict
from markdown import markdown
from bs4 import BeautifulSoup
from pypdf import PdfReader
from app.confluence_importer import ConfluenceImporter


def fixed_size_text_chunking(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    split_text = [text[i: i+chunk_size] if i == 0 else text[i - overlap: i + chunk_size - overlap]
                  for i in np.arange(0, len(text) + overlap, chunk_size)]

    return split_text


class DocumentFormats(Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    CONFLUENCE = "confluence"


class DocumentExtractor(BaseModel, ABC):

    @classmethod
    @abstractmethod
    def extract(cls, document: bytes):
        raise NotImplementedError

    @classmethod
    def fixed_size_text_chunking(cls, text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
        split_text = [text[i: i + chunk_size] if i == 0 else text[i - overlap: i + chunk_size - overlap]
                      for i in np.arange(0, len(text) + overlap, chunk_size)]

        return split_text


class MarkdownExtractor(DocumentExtractor):

    @classmethod
    def extract(cls, document: bytes) -> str:
        html = markdown(document)
        text = "".join(BeautifulSoup(html, features="html.parser").find_all(text=True))

        return text


class PdfExtractor(DocumentExtractor):

    @classmethod
    def extract(cls, document: bytes):
        reader = PdfReader(io.BytesIO(document))
        text = "".join([page.extract_text() for page in reader.pages])

        return text


class ConfluenceExtractor(DocumentExtractor):
    confluence_settings: Dict

    def extract(self, document: bytes):
        exporter = ConfluenceImporter(self.confluence_settings)
        exporter.run()
