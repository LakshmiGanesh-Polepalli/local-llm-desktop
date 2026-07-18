import io
import math
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document

class DocumentProcessor:
    @staticmethod
    def extract_text_from_bytes(file_bytes: bytes, file_extension: str) -> str:
        """Extracts raw text from PDF or DOCX binary streams."""
        text = ""
        ext = file_extension.lower().replace(".", "")
        
        if ext == "pdf":
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        elif ext in ["docx", "doc"]:
            docx_file = io.BytesIO(file_bytes)
            doc = Document(docx_file)
            for paragraph in doc.paragraphs:
                if paragraph.text:
                    text += paragraph.text + "\n"
        elif ext in ["txt", "md", "csv", "json", "py", "cpp", "h"]:
            text = file_bytes.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file format: .{ext}")
            
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Splits text using a lightweight sliding window token/character strategy.
        Returns data ready for embedding matrix layout.
        """
        words = text.split()
        chunks = []
        
        if not words:
            return chunks
            
        stride = chunk_size - chunk_overlap
        num_chunks = max(1, math.ceil((len(words) - chunk_overlap) / stride)) if len(words) > chunk_size else 1
        
        for i in range(num_chunks):
            start_idx = i * stride
            end_idx = start_idx + chunk_size
            chunk_words = words[start_idx:end_idx]
            
            chunks.append({
                "text": " ".join(chunk_words),
                "meta": {"word_start": start_idx, "word_end": start_idx + len(chunk_words)}
            })
            
        return chunks
