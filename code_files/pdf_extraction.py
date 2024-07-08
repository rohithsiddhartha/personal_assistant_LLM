import fitz  # PyMuPDF
import pdfplumber
import os
import pandas as pd

class PDFExtraction:
    def __init__(self, pdf_path, base_extraction_dir='pdf_extractions'):
        self.pdf_path = pdf_path
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.base_dir = os.path.join(base_extraction_dir, self.pdf_name)
        self.text_dir = os.path.join(self.base_dir, "text")
        self.image_dir = os.path.join(self.base_dir, "images")
        self.table_dir = os.path.join(self.base_dir, "tables")
        self.text_file = os.path.join(self.text_dir, f"{self.pdf_name}.txt")
        
        # Create necessary directories
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        with open(self.text_file, "w", encoding="utf-8") as file:
            file.write(text)
        
        print(f"Text extracted to file: {self.text_file}")

    def extract_images(self):
        doc = fitz.open(self.pdf_path)
        for page_number in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_number)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(self.image_dir, f"page{page_number+1}_img{img_index+1}.{image_ext}")
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
        
        print(f"Images extracted to directory: {self.image_dir}")

    def extract_tables(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_path = os.path.join(self.table_dir, f"table_page{i+1}_{j+1}.csv")
                        df.to_csv(table_path, index=False)
        
        print(f"Tables extracted to directory: {self.table_dir}")

    def extract_all(self):
        self.extract_text()
        self.extract_images()
        self.extract_tables()

# Usage example
pdf_path = "pranay.pdf"  # Path to the uploaded PDF
pdf_extractor = PDFExtraction(pdf_path, base_extraction_dir='pdf_extractions')
pdf_extractor.extract_all()
