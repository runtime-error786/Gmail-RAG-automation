import os
import fitz 
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser, JSONParser
from rest_framework import status
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import time

os.environ['PINECONE_API_KEY'] = ''
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag"
index = pinecone.Index(index_name)

llm_text = Ollama(model="llama3.1")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_model, namespace="real")

EMAIL = "mustafa782a@gmail.com"
PASSWORD = ""
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def extract_pdf_text(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    document.close()
    return text

def fetch_unread_email():
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, 'UNSEEN')
    if status != 'OK' or not messages[0]:
        return None, None

    email_id = messages[0].split()[-1]
    status, msg_data = mail.fetch(email_id, "(RFC822)")
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    sender = msg["From"]
    subject = msg["Subject"]
    content = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                content += part.get_payload(decode=True).decode()
    else:
        content = msg.get_payload(decode=True).decode()
    return content, sender

def send_email_response(recipient, subject, message):
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL, PASSWORD)
    msg = MIMEText(message)
    msg["From"] = EMAIL
    msg["To"] = recipient
    msg["Subject"] = subject
    server.sendmail(EMAIL, recipient, msg.as_string())
    server.quit()

class PDFUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        pdf_file = request.FILES['pdf']
        pdf_path = f'{pdf_file.name}'

        with open(pdf_path, 'wb') as f:
            f.write(pdf_file.read())

        extracted_text = extract_pdf_text(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = text_splitter.split_text(extracted_text)

        for i, chunk in enumerate(chunks):
            query_result = embedding_model.embed_query(chunk)
            index.upsert(
                vectors=[{
                    "id": str(i),
                    "values": query_result,
                    "metadata": {"text": chunk}
                }],
                namespace="real"
            )

        return Response({"message": "PDF processed and embeddings stored."}, status=status.HTTP_201_CREATED)

@api_view(["POST"])
def query_view(request):
    poll_interval = 10  

    while True:
        content, sender = fetch_unread_email()

        if content and sender:
            print(f"New email from {sender}: {content}")

            retrieval_qa = RetrievalQA.from_chain_type(
                llm=llm_text,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(k=3)
            )
            response_text = retrieval_qa.run(content)

            send_email_response(sender, "Automated Response", response_text)

            

        print("No new emails found, checking again...")
        time.sleep(poll_interval)  
