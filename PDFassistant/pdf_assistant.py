import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2, SearchType
from phi.llm.groq import Groq
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder 


import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

embedder = SentenceTransformerEmbedder(batch_size=100, dimensions=768, model="sentence-transformers/all-mpnet-base-v2" )

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url, embedder=embedder)
)

sample_embedding = embedder.get_embedding(text="This is a test sentence.")
print(f"Sample embedding dimensions: {len(sample_embedding)}")

knowledge_base.load()

storage = PgAssistantStorage(db_url=db_url, table_name="pdf_assistant")

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids()
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id, 
        user_id=user,
        knowledge_base=knowledge_base, 
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
        llm= Groq(model="llama-3.3-70b-versatile", name="Groq", embedder=embedder),
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Assistant run: {run_id}\n")
    
    else:
        print(f"Continuing Assistant run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)