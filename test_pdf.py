from pypdf import PdfReader

pdf_path = "data/medical_documents/sick-with-the-flu.pdf"

reader = PdfReader(pdf_path)

for page in reader.pages:
    text = page.extract_text()
    print(text)