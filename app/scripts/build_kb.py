from app.rag import build_vector_store


if __name__ == "__main__":
    store = build_vector_store(force=True)
    print(f"Indexed chunks: {len(store.metadata)}")
