import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import io
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
import numpy as np
import time

class CLIPEmbeddings(Embeddings):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

class RAGPipeline:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", llm_model_name="gpt-4o-mini", llm_provider="openai"):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        self.clip_model_name = clip_model_name
        self.llm_model_name = llm_model_name
        self.llm_provider = llm_provider

        self.clip_model = None
        self.clip_processor = None
        self.clip_embeddings = None
        self.llm = None
        
        self.vector_store = None
        self.all_docs = []
        self.image_data_store = {}
        self.image_captions = {}
        self.table_data_store = {}
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(3)  # Max 3 concurrent API calls
        self.last_api_call = 0
        self.min_delay = 0.5  # Minimum 500ms between calls

    def _initialize_models(self):
        """Initializes the models and embeddings."""
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name, use_safetensors=True)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model.eval()
            self.clip_embeddings = CLIPEmbeddings(self.clip_model, self.clip_processor)
        
        if self.llm is None:
            self.llm = init_chat_model(self.llm_model_name, model_provider=self.llm_provider)

    async def embed_image(self, image_data):
        """Embeds image data using the CLIP model."""
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = image_data
            
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

    async def embed_text(self, text):
        """Embeds text using the CLIP model."""
        embedding = self.clip_embeddings.embed_query(text)
        return np.array(embedding)  # Convert to numpy array for consistency

    async def generate_image_caption(self, pil_image, context_text="", retry_count=0):
        """Generates a caption for an image using GPT-4o-mini vision with rate limiting."""
        async with self.semaphore:
            try:
                # Rate limiting delay
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call
                if time_since_last_call < self.min_delay:
                    await asyncio.sleep(self.min_delay - time_since_last_call)
                
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                prompt = "Describe this image concisely in 2-3 sentences. Focus on the main subjects, objects, and key visual elements."
                if context_text:
                    prompt += f"\n\nSurrounding text context: {context_text[:300]}"
                
                messages = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            }
                        ]
                    )
                ]
                
                self.last_api_call = time.time()
                response = await self.llm.ainvoke(messages)
                return response.content
                
            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() and retry_count < 3:
                    # Extract wait time from error message if available
                    wait_time = 2 ** retry_count  # Exponential backoff: 1s, 2s, 4s
                    print(f"Rate limit hit, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    return await self.generate_image_caption(pil_image, context_text, retry_count + 1)
                else:
                    print(f"Error generating caption (attempt {retry_count + 1}): {e}")
                    # Return a basic description based on context if available
                    if context_text:
                        return f"Image related to: {context_text[:200]}"
                    return "Image content (caption generation failed)"

    def extract_table_from_page(self, page):
        """Extracts tables from a PDF page."""
        tables = []
        try:
            tabs = page.find_tables()
            if tabs.tables:
                for i, table in enumerate(tabs.tables):
                    df = table.to_pandas()
                    if df is not None and not df.empty:
                        table_text = df.to_markdown(index=False)
                        tables.append({
                            'index': i,
                            'content': table_text,
                            'bbox': table.bbox
                        })
        except Exception as e:
            print(f"Error extracting tables: {e}")
        
        return tables

    def get_text_around_bbox(self, page, bbox, margin=100):
        """Gets text around a bounding box (for image/table context)."""
        x0, y0, x1, y1 = bbox
        expanded_bbox = (
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(page.rect.width, x1 + margin),
            min(page.rect.height, y1 + margin)
        )
        
        blocks = page.get_text("blocks", clip=expanded_bbox)
        context_text = ""
        for block in blocks:
            if block[6] == 0:  # Text block
                block_bbox = block[:4]
                if not self._bbox_overlap(bbox, block_bbox):
                    context_text += block[4] + " "
        
        return context_text.strip()
    
    def extract_figure_number(self, text):
        """Extract figure number from text like 'Figure 9.' or 'Fig. 9'"""
        import re
        patterns = [
            r'Figure\s+(\d+)',
            r'Fig\.\s+(\d+)',
            r'FIGURE\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def find_figure_references_in_text(self, page_text, figure_num):
        """Find text that references a specific figure number."""
        import re
        # Look for mentions of the figure in the text
        patterns = [
            rf'[^.]*Figure\s+{figure_num}[^.]*\.',
            rf'[^.]*Fig\.\s+{figure_num}[^.]*\.',
            rf'[^.]*illustrated in Figure\s+{figure_num}[^.]*\.',
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE | re.DOTALL)
            references.extend(matches)
        
        return ' '.join(references).strip()

    def _bbox_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap."""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)

    async def _process_page(self, page, page_num, splitter, full_doc_text=""):
        """Processes a single page of a PDF."""
        docs, embeddings, image_data, image_captions, table_data = [], [], {}, {}, {}

        full_page_text = page.get_text()
        
        # Process Tables
        tables = self.extract_table_from_page(page)
        for table_info in tables:
            table_id = f"page_{page_num}_table_{table_info['index']}"
            table_content = table_info['content']
            context = self.get_text_around_bbox(page, table_info['bbox'])
            table_data[table_id] = table_content
            
            table_doc = Document(
                page_content=f"Table Context: {context}\n\nTable Content:\n{table_content}",
                metadata={"page": page_num, "type": "table", "table_id": table_id}
            )
            
            embedding = await self.embed_text(table_doc.page_content)
            embeddings.append(embedding)
            docs.append(table_doc)

        # Process Text
        if full_page_text.strip():
            temp_doc = Document(page_content=full_page_text, metadata={"page": page_num, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            
            for chunk in text_chunks:
                embedding = await self.embed_text(chunk.page_content)
                embeddings.append(embedding)
                docs.append(chunk)
        
        # Process Images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Skip very small images
                if pil_image.width < 50 or pil_image.height < 50:
                    continue
                
                image_id = f"page_{page_num}_img_{img_index}"
                
                # Store image as base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data[image_id] = img_base64
                
                # Get surrounding context (caption and nearby text)
                img_rects = page.get_image_rects(xref)
                context_text = ""
                figure_number = None
                
                if img_rects:
                    img_bbox = img_rects[0]
                    context_text = self.get_text_around_bbox(page, img_bbox, margin=150)
                    
                    # Try to extract figure number from nearby text
                    figure_number = self.extract_figure_number(context_text)
                
                # If we found a figure number, search the entire document for references
                figure_references = ""
                if figure_number:
                    # Search current page
                    figure_references = self.find_figure_references_in_text(full_page_text, figure_number)
                    
                    # Also search nearby pages if available
                    if full_doc_text:
                        doc_references = self.find_figure_references_in_text(full_doc_text, figure_number)
                        if doc_references and doc_references not in figure_references:
                            figure_references += " " + doc_references
                
                # Generate caption with rate limiting
                caption = await self.generate_image_caption(pil_image, context_text)
                image_captions[image_id] = caption
                
                # Create enriched content with figure info
                combined_content = f"Image Description: {caption}"
                
                if figure_number:
                    combined_content = f"Figure {figure_number}. {combined_content}"
                
                if context_text:
                    combined_content += f"\n\nCaption/Context: {context_text}"
                
                if figure_references:
                    combined_content += f"\n\nDocument References: {figure_references}"
                
                # Get embeddings
                text_embedding = await self.embed_text(combined_content)
                visual_embedding = await self.embed_image(pil_image)
                
                # Ensure both are numpy arrays and combine
                text_embedding = np.array(text_embedding) if not isinstance(text_embedding, np.ndarray) else text_embedding
                visual_embedding = np.array(visual_embedding) if not isinstance(visual_embedding, np.ndarray) else visual_embedding
                
                combined_embedding = 0.6 * text_embedding + 0.4 * visual_embedding
                embeddings.append(combined_embedding)
                
                image_doc = Document(
                    page_content=combined_content,
                    metadata={
                        "page": page_num, 
                        "type": "image", 
                        "image_id": image_id,
                        "caption": caption,
                        "figure_number": figure_number
                    }
                )
                docs.append(image_doc)
                
            except Exception as e:
                print(f"Error processing image {img_index} on page {page_num}: {e}")
                continue
        
        return docs, embeddings, image_data, image_captions, table_data

    async def process_pdf(self, pdf_path):
        """Processes the PDF and creates the vector store."""
        self._initialize_models()
        
        doc = fitz.open(pdf_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Process pages sequentially to better manage rate limits
        print(f"Processing {len(doc)} pages...")
        results = []
        for i, page in enumerate(doc):
            print(f"Processing page {i+1}/{len(doc)}...")
            result = await self._process_page(page, i, splitter)
            results.append(result)
            # Small delay between pages to manage rate limits
            if i < len(doc) - 1:
                await asyncio.sleep(0.3)
        
        doc.close()
        
        self.all_docs = []
        all_embeddings = []
        self.image_data_store = {}
        self.image_captions = {}
        self.table_data_store = {}
        
        for docs, embeddings, image_data, image_captions, table_data in results:
            self.all_docs.extend(docs)
            all_embeddings.extend(embeddings)
            self.image_data_store.update(image_data)
            self.image_captions.update(image_captions)
            self.table_data_store.update(table_data)
            
        if not self.all_docs:
            return

        # Convert embeddings to proper format for FAISS
        text_embeddings = [
            (doc.page_content, emb.tolist() if isinstance(emb, np.ndarray) else emb) 
            for doc, emb in zip(self.all_docs, all_embeddings)
        ]
        
        self.vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=self.clip_embeddings
        )
        
        print(f"✓ Processed {len(self.all_docs)} documents successfully!")

    async def ask_question(self, question):
        """Asks a question to the LLM with the context of the PDF."""
        if not self.vector_store:
            return "Please upload a PDF first."

        self._initialize_models()

        # Embed the question
        query_embedding = await self.embed_text(question)
        
        # Convert to list for FAISS
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search_by_vector(query_embedding_list, k=7)
        
        # Format the context
        context_parts = []
        images_to_include = []
        
        for i, doc in enumerate(retrieved_docs):
            if doc.metadata.get("type") == "image":
                image_id = doc.metadata.get("image_id")
                caption = doc.metadata.get("caption", "No caption available")
                context_parts.append(f"[Image {i+1}]: {caption}")
                if image_id in self.image_data_store:
                    images_to_include.append(image_id)
            elif doc.metadata.get("type") == "table":
                table_id = doc.metadata.get("table_id")
                context_parts.append(f"[Table {i+1}]:\n{doc.page_content}")
            else:
                context_parts.append(f"[Text {i+1}]: {doc.page_content}")
        
        context = "\n\n".join(context_parts)

        # Create the prompt
        prompt_template = """You are a helpful assistant analyzing a PDF document. Use the provided context to answer the question accurately and comprehensively.

The context includes:
- Text excerpts from the document
- Descriptions and captions of images
- Tables with their surrounding context

When referencing images or tables, refer to them by their numbers (e.g., "As shown in Image 1..." or "According to Table 2...").

Context:
{context}

Question: {question}

Provide a detailed and accurate answer based on the context above:"""
        
        prompt_text = prompt_template.format(context=context, question=question)
        
        # Create the message content
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add actual images to the message for visual verification
        for image_id in images_to_include[:3]:  # Limit to 3 images
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self.image_data_store[image_id]}",
                    "detail": "high"
                }
            })
        
        messages = [HumanMessage(content=message_content)]

        # Get the response from the LLM
        response = await self.llm.ainvoke(messages)
        return response.contentq    